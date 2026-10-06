//! Credential storage for Smartloop authentication.
//!
//! Port of `smartloop/auth/credential_store.py`.  Uses a local
//! Fernet-encrypted file store at `SLP_HOME/.credentials` with the encryption
//! key kept in `SLP_HOME/.key`.
//!
//! Resolution order for developer tokens:
//!   1. `-t` flag (stored via [`set_token`] before any API calls)
//!   2. File-based credential store
//!   3. `SLP_DEVELOPER_TOKEN` environment variable

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use parking_lot::{Mutex, ReentrantMutex, RwLock};
use serde_json::{Map, Value};

use crate::paths::{expand_home, real_home};

const DEVELOPER_TOKEN: &str = "developer_token";
const ACCESS_TOKEN: &str = "access_token";
const REFRESH_TOKEN: &str = "refresh_token";

/// The decrypted credential file: a flat JSON object.
pub type Creds = Map<String, Value>;

/// Held across the read-modify-write in `set` and `clear`.  The whole file is
/// rewritten to change one key, so two unsynchronized writers — the token
/// refresh agent storing an access token while a request thread stores the
/// rotated refresh token — would each save a copy of the file as it looked
/// before the other's change, dropping one of the two silently.
///
/// Process-local.  A second process (the CLI alongside a running server) can
/// still interleave; [`save_creds`] keeps that from corrupting the file, but
/// the lost-update window remains and wants a real file lock.
///
/// The supervisor's agent children are deliberately not in that set.  They
/// read this store constantly and never write it: the one write they could
/// reach, redeeming the refresh token, is declined in a pinned process.
/// Readers take no lock at all and do not need one — [`save_creds`] renames a
/// finished file into place, so a reader sees one whole version or the other.
static WRITE_LOCK: Lazy<ReentrantMutex<()>> = Lazy::new(|| ReentrantMutex::new(()));

/// Test seam standing in for monkeypatching `_save_creds`: when set, every
/// call to [`save_creds`] first passes the creds to this observer.
pub type SaveObserver = Box<dyn Fn(&Creds) + Send + Sync>;
static SAVE_OBSERVER: Lazy<Mutex<Option<SaveObserver>>> = Lazy::new(|| Mutex::new(None));

/// Install (or with `None`, remove) the [`save_creds`] observer.
pub fn set_save_observer(observer: Option<SaveObserver>) {
    *SAVE_OBSERVER.lock() = observer;
}

/// Test seam standing in for monkeypatching `_slp_home`: when set, the store
/// lives in this directory instead of `SLP_HOME` / `~/.smartloop`.
static SLP_HOME_OVERRIDE: Lazy<RwLock<Option<PathBuf>>> = Lazy::new(|| RwLock::new(None));

/// Redirect the credential store to `dir` (or, with `None`, restore the normal
/// `SLP_HOME` / `~/.smartloop` resolution).
pub fn set_slp_home_override(dir: Option<PathBuf>) {
    *SLP_HOME_OVERRIDE.write() = dir;
}

// -- file-based storage ----------------------------------------------------

/// `SLP_HOME` (tilde-expanded) or `<real home>/.smartloop`.
pub fn slp_home() -> PathBuf {
    if let Some(dir) = SLP_HOME_OVERRIDE.read().clone() {
        return dir;
    }
    match std::env::var("SLP_HOME") {
        Ok(home) if !home.is_empty() => PathBuf::from(expand_home(&home)),
        _ => PathBuf::from(real_home()).join(".smartloop"),
    }
}

/// `<slp_home>/.key`
pub fn key_path() -> PathBuf {
    slp_home().join(".key")
}

/// `<slp_home>/.credentials`
pub fn creds_path() -> PathBuf {
    slp_home().join(".credentials")
}

/// Create the directory if needed and lock permissions to owner-only (700).
fn ensure_dir(path: &Path) -> std::io::Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Open `path` for writing with owner-only (0600) permissions.
fn open_private(path: &Path, create_new: bool) -> std::io::Result<fs::File> {
    let mut opts = OpenOptions::new();
    opts.write(true);
    if create_new {
        opts.create_new(true);
    } else {
        opts.create(true).truncate(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
}

/// Return a Fernet instance, generating the key file if needed.
///
/// The key is created with `O_EXCL` and never truncated in place.  Truncating
/// was the same hazard [`save_creds`] was hardened against, and worse here:
/// this runs from [`load_creds`], which readers reach *without* the write
/// lock, so a reader arriving mid-write got a short key, `Fernet` failed, and
/// the catch-all in `load_creds` reported it as "no credentials" — a phantom
/// logout.
///
/// Losing the `O_EXCL` race means another writer created the key first; read
/// theirs rather than overwriting it, which would strand every credential
/// already encrypted under it.
fn get_fernet() -> anyhow::Result<fernet::Fernet> {
    let kp = key_path();
    if !kp.exists() {
        if let Some(parent) = kp.parent() {
            ensure_dir(parent)?;
        }
        match open_private(&kp, true) {
            Ok(mut f) => {
                f.write_all(fernet::Fernet::generate_key().as_bytes())?;
                f.flush()?;
                f.sync_all()?;
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.into()),
        }
    }

    let mut key = String::new();
    fs::File::open(&kp)?.read_to_string(&mut key)?;
    fernet::Fernet::new(key.trim()).ok_or_else(|| anyhow::anyhow!("Fernet key must be 32 url-safe base64-encoded bytes."))
}

/// Load and decrypt the credential file; any failure reads as `{}`.
pub fn load_creds() -> Creds {
    let cp = creds_path();
    if !cp.exists() {
        return Creds::new();
    }
    let attempt = || -> anyhow::Result<Creds> {
        let fernet = get_fernet()?;
        let token = fs::read_to_string(&cp)?;
        let plain = fernet
            .decrypt(token.trim())
            .map_err(|_| anyhow::anyhow!("invalid credential token"))?;
        match serde_json::from_slice::<Value>(&plain)? {
            Value::Object(map) => Ok(map),
            _ => Err(anyhow::anyhow!("credential file is not a JSON object")),
        }
    };
    attempt().unwrap_or_default()
}

/// Write the credential file, replacing it in one step.
///
/// Writing in place truncated the file before the ciphertext landed, so a
/// reader arriving mid-write — or a process that died there — found a partial
/// Fernet blob.  [`load_creds`] cannot tell that apart from a bad key and
/// answers `{}` either way, which presents as a silent, total logout.  Write a
/// sibling and rename: readers see the whole of one version or the whole of
/// the other.
pub fn save_creds(creds: &Creds) -> anyhow::Result<()> {
    if let Some(observer) = SAVE_OBSERVER.lock().as_ref() {
        observer(creds);
    }
    let fernet = get_fernet()?;
    let cp = creds_path();
    if let Some(parent) = cp.parent() {
        ensure_dir(parent)?;
    }
    let tmp = PathBuf::from(format!("{}.{}.tmp", cp.display(), std::process::id()));
    let write = || -> anyhow::Result<()> {
        let payload = serde_json::to_vec(&Value::Object(creds.clone()))?;
        let mut f = open_private(&tmp, false)?;
        f.write_all(fernet.encrypt(&payload).as_bytes())?;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, &cp)?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

// -- helpers ---------------------------------------------------------------

/// The stored value for `key` as a string (`None` when absent or null).
fn get(key: &str) -> Option<String> {
    match load_creds().get(key) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => Some(other.to_string()),
    }
}

/// Python truthiness of an optional string: `None` and `""` are both falsy.
fn truthy(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}

fn set(key: &str, value: &str) {
    let _guard = WRITE_LOCK.lock();
    let mut creds = load_creds();
    creds.insert(key.to_string(), Value::String(value.to_string()));
    if let Err(e) = save_creds(&creds) {
        tracing::error!(target: "smartloop.auth.credential_store", "Failed to save credentials: {e}");
    }
}

fn clear(key: &str) {
    let _guard = WRITE_LOCK.lock();
    let mut creds = load_creds();
    creds.shift_remove(key);
    if let Err(e) = save_creds(&creds) {
        tracing::error!(target: "smartloop.auth.credential_store", "Failed to save credentials: {e}");
    }
}

// -- developer token -------------------------------------------------------

/// Retrieve the developer token: file store first, then `SLP_DEVELOPER_TOKEN`.
pub fn get_token() -> Option<String> {
    truthy(get(DEVELOPER_TOKEN)).or_else(|| std::env::var("SLP_DEVELOPER_TOKEN").ok())
}

/// Store a developer token.
pub fn set_token(token: &str) {
    set(DEVELOPER_TOKEN, token);
}

/// Remove the stored developer token.
pub fn clear_token() {
    clear(DEVELOPER_TOKEN);
}

// -- JWT access token (set by the Electron client after OAuth) -------------

/// The stored OAuth access token.
pub fn get_access_token() -> Option<String> {
    get(ACCESS_TOKEN)
}

/// Store the OAuth access token.
pub fn set_access_token(token: &str) {
    set(ACCESS_TOKEN, token);
}

/// Remove the stored OAuth access token.
pub fn clear_access_token() {
    clear(ACCESS_TOKEN);
}

// -- refresh token ---------------------------------------------------------

/// The stored OAuth refresh token.
pub fn get_refresh_token() -> Option<String> {
    get(REFRESH_TOKEN)
}

/// Store the OAuth refresh token.
pub fn set_refresh_token(token: &str) {
    set(REFRESH_TOKEN, token);
}

/// Remove the stored OAuth refresh token.
pub fn clear_refresh_token() {
    clear(REFRESH_TOKEN);
}

// -- the OAuth pair, written as one --------------------------------------

/// Store the access token, and the rotated refresh token with it.
///
/// One file version for the pair.  Storing them with two calls leaves a moment
/// where the new access token is on disk beside the refresh token that was
/// just spent to mint it, and a crash or a kill in that moment makes the state
/// permanent: the next redemption presents a spent token, is refused, and
/// reads the refusal as "re-login required".  A single-use refresh token makes
/// that window unrecoverable rather than merely untidy.
///
/// `refresh` is optional because the endpoint does not always rotate; `None`
/// (or empty) leaves the stored one alone.
pub fn set_tokens(access: &str, refresh: Option<&str>) {
    let _guard = WRITE_LOCK.lock();
    let mut creds = load_creds();
    creds.insert(ACCESS_TOKEN.to_string(), Value::String(access.to_string()));
    if let Some(refresh) = refresh.filter(|r| !r.is_empty()) {
        creds.insert(REFRESH_TOKEN.to_string(), Value::String(refresh.to_string()));
    }
    if let Err(e) = save_creds(&creds) {
        tracing::error!(target: "smartloop.auth.credential_store", "Failed to save credentials: {e}");
    }
}

/// Drop both OAuth credentials in one file version.
///
/// The counterpart to [`set_tokens`], for the refusal that ends the session:
/// clearing them one at a time leaves a readable state holding a refresh token
/// whose access token is already gone.
pub fn clear_tokens() {
    let _guard = WRITE_LOCK.lock();
    let mut creds = load_creds();
    creds.shift_remove(ACCESS_TOKEN);
    creds.shift_remove(REFRESH_TOKEN);
    if let Err(e) = save_creds(&creds) {
        tracing::error!(target: "smartloop.auth.credential_store", "Failed to save credentials: {e}");
    }
}

// -- unified resolver ------------------------------------------------------

/// Return the best available auth token: access_token > developer_token.
pub fn get_active_token() -> Option<String> {
    truthy(get(ACCESS_TOKEN)).or_else(get_token)
}

/// True if any auth token is available.
pub fn has_credentials() -> bool {
    !auth_headers().is_empty()
}

/// Return the correct HTTP auth headers based on available credentials.
///
/// - access token → `{"Authorization": "Bearer <token>"}`
/// - developer token → `{"X-Api-Key": "<token>"}`
/// - neither → `{}`
///
/// The developer token comes from [`get_token`], so `SLP_DEVELOPER_TOKEN`
/// counts here exactly as the module docs say it does.  Reading the file store
/// directly meant an env-var-only user was reported as signed in by
/// `/v1/auth/status` — which asks [`get_token`] — while every request went out
/// with no credentials at all.
pub fn auth_headers() -> BTreeMap<String, String> {
    let mut headers = BTreeMap::new();
    if let Some(access) = truthy(get(ACCESS_TOKEN)) {
        headers.insert("Authorization".to_string(), format!("Bearer {access}"));
        return headers;
    }
    if let Some(dev) = truthy(get_token()) {
        headers.insert("X-Api-Key".to_string(), dev);
    }
    headers
}
