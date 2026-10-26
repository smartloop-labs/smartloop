//! Where the user's home actually is, once a store's sandbox has moved it.
//!
//! Port of `smartloop/paths.py`.
//!
//! `$HOME` is what every tilde expands against, and both stores we ship
//! through rewrite it to a private per-app directory:
//!
//! * the macOS App Sandbox (Mac App Store) points it at
//!   `~/Library/Containers/<bundle-id>/Data`
//! * snapd, under strict confinement, points it at
//!   `~/snap/<snap-name>/<revision>`
//!
//! Neither is a directory the user has ever put a file in, so anything they
//! wrote a `~` into — a project directory, an MCP server's `command`, a
//! document handed to a parser — expanded to a path that was never going to
//! exist, and the framework home itself landed inside the container instead of
//! at `~/.smartloop`.
//!
//! What the sandboxes rewrite is the environment variable, not the passwd
//! database, so the record behind `getpwuid` still names the real home; snapd
//! 2.46 and later also states it outright in `SNAP_REAL_HOME`.  That is the
//! whole of the trick here.
//!
//! Resolving the path is not the same as being allowed to read it: under the
//! App Sandbox the user has to pick the directory, and a strict snap reaches
//! the real home only through the `home` interface.  This module answers
//! *where*, and the caller still has to cope with a denial.

use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;

/// How macOS spells the App Sandbox container it rewrites `$HOME` to.
pub const SANDBOX_HOME_MARKER: &str = "/Library/Containers/";

/// How snapd spells `$SNAP_USER_DATA`, which it rewrites `$HOME` to:
/// `<real home>/snap/<snap-name>/<revision>`.
pub const SNAP_HOME_RE: &str = r"^(?P<home>.+)/snap/[^/]+/[^/]+/?$";

static SNAP_HOME_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(SNAP_HOME_RE).expect("valid regex"));

/// `os.environ.get(name)`, treating a non-UTF-8 value as absent.
fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// The real home `home` is a sandbox directory inside, or `None`.
///
/// `None` means `$HOME` is not a confinement directory at all and so is to be
/// taken at its word — a service unit or a test that sets it means it, and
/// only the two spellings below are ever overridden.
fn unconfined_parent(home: &str) -> Option<String> {
    if let Some(index) = home.find(SANDBOX_HOME_MARKER) {
        if index > 0 {
            return Some(home[..index].to_string());
        }
    }
    // A plain `~/snap/...` is a perfectly ordinary directory to have; it is
    // only the rewritten home when we are in fact running inside a snap, which
    // snapd says by setting `SNAP_NAME`.
    if env_var("SNAP_NAME").is_some_and(|v| !v.is_empty()) {
        if let Some(caps) = SNAP_HOME_REGEX.captures(home) {
            return Some(caps["home"].to_string());
        }
    }
    None
}

#[cfg(unix)]
#[derive(Clone, Copy)]
enum PasswdKey<'a> {
    Uid(libc::uid_t),
    Name(&'a str),
}

/// Read a passwd entry's home directory through `getpwuid_r`/`getpwnam_r`.
#[cfg(unix)]
fn passwd_lookup(by: PasswdKey<'_>) -> Option<String> {
    use std::ffi::{CStr, CString};

    let mut buf_len: usize = 4096;
    loop {
        let mut buf = vec![0 as libc::c_char; buf_len];
        // SAFETY: `passwd` is a plain C struct; all-zero is a valid value and
        // it is only read after the libc call reports success.
        let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
        let mut result: *mut libc::passwd = std::ptr::null_mut();
        let rc = match by {
            // SAFETY: every pointer refers to a live, correctly sized buffer.
            PasswdKey::Uid(uid) => unsafe {
                libc::getpwuid_r(uid, &mut pwd, buf.as_mut_ptr(), buf.len(), &mut result)
            },
            PasswdKey::Name(name) => {
                let cname = CString::new(name).ok()?;
                // SAFETY: as above; `cname` outlives the call.
                unsafe { libc::getpwnam_r(cname.as_ptr(), &mut pwd, buf.as_mut_ptr(), buf.len(), &mut result) }
            }
        };
        if rc == libc::ERANGE && buf_len < (1 << 20) {
            buf_len *= 2;
            continue;
        }
        if rc != 0 || result.is_null() || pwd.pw_dir.is_null() {
            return None;
        }
        // SAFETY: `pw_dir` points into `buf`, which is still alive.
        let dir = unsafe { CStr::from_ptr(pwd.pw_dir) };
        return Some(dir.to_string_lossy().into_owned());
    }
}

/// The passwd record's home directory for the current uid, if readable.
#[cfg(unix)]
pub fn passwd_home() -> Option<String> {
    // SAFETY: getuid has no preconditions and cannot fail.
    passwd_lookup(PasswdKey::Uid(unsafe { libc::getuid() }))
}

/// The passwd record's home directory for the current uid, if readable.
/// There is no passwd database off Unix.
#[cfg(not(unix))]
pub fn passwd_home() -> Option<String> {
    None
}

/// Home directory of the named user (`pwd.getpwnam(name).pw_dir`).
#[cfg(unix)]
fn passwd_home_of(name: &str) -> Option<String> {
    passwd_lookup(PasswdKey::Name(name))
}

#[cfg(not(unix))]
fn passwd_home_of(_name: &str) -> Option<String> {
    None
}

/// The user's home, reading past a store sandbox's rewrite of `$HOME`.
pub fn real_home() -> String {
    let home = env_var("HOME").unwrap_or_default();
    let Some(parent) = unconfined_parent(&home) else {
        // Not confined, so `$HOME` is the answer and is taken as given.
        return expanduser("~");
    };

    // snapd >= 2.46 states the real home outright.  Only consulted once
    // `$HOME` has shown we are the confined process: a snap that launches
    // something else — a terminal emulator starting a shell, say — leaks its
    // whole `SNAP_*` block into a child that is not confined at all, and that
    // child's `$HOME` means exactly what it says.
    if let Some(snap_home) = env_var("SNAP_REAL_HOME") {
        if !snap_home.is_empty() {
            return snap_home;
        }
    }

    // Neither sandbox rewrites the passwd record.  With no passwd record to
    // read, the directory the confinement path sits in is the next best answer.
    match passwd_home() {
        Some(dir) if !dir.is_empty() => dir,
        _ => parent,
    }
}

/// The sandbox directory `$HOME` was rewritten to, or `None`.
///
/// The counterpart to [`real_home`]: where the framework *used* to put its
/// state on a store build, and so where an install made before the home was
/// resolved properly still has it.  `None` off a store build, which is every
/// developer machine and the Developer ID build.
pub fn confined_home() -> Option<String> {
    let home = env_var("HOME").unwrap_or_default();
    if unconfined_parent(&home).is_some() {
        Some(home)
    } else {
        None
    }
}

/// `~`-expand `path` against [`real_home`].
///
/// `~otheruser/...` names somebody else's home and is left to
/// [`expanduser`], which is the only thing that knows how to resolve it.
pub fn expand_home(path: &str) -> String {
    expand_home_with(path, real_home)
}

/// [`expand_home`] with an injectable real-home resolver (test seam standing
/// in for monkeypatching `smartloop.paths.real_home`).
pub fn expand_home_with(path: &str, real_home: impl Fn() -> String) -> String {
    if !path.starts_with('~') {
        return path.to_string();
    }
    if path == "~" {
        return real_home();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return path_join(&real_home(), rest);
    }
    expanduser(path)
}

/// `os.path.join(a, b)` for two POSIX components.
fn path_join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        return b.to_string();
    }
    if a.is_empty() || a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

/// `os.path.expanduser` equivalent: `~` / `~/x` against `$HOME` (falling
/// back to passwd), `~user/x` against that user's passwd record; anything
/// unresolvable is returned unchanged.
pub fn expanduser(path: &str) -> String {
    if !path.starts_with('~') {
        return path.to_string();
    }
    let i = path[1..].find('/').map(|p| p + 1).unwrap_or(path.len());
    let userhome = if i == 1 {
        match std::env::var_os("HOME") {
            Some(home) => home.to_string_lossy().into_owned(),
            None => match passwd_home() {
                Some(dir) => dir,
                // No passwd entry for the current uid: leave the path alone.
                None => return path.to_string(),
            },
        }
    } else {
        match passwd_home_of(&path[1..i]) {
            Some(dir) => dir,
            None => return path.to_string(),
        }
    };
    let userhome = userhome.trim_end_matches('/');
    let expanded = format!("{userhome}{}", &path[i..]);
    if expanded.is_empty() {
        "/".to_string()
    } else {
        expanded
    }
}

/// `path`, `~`-expanded and canonicalized, when it names a folder this
/// process can list; `None` otherwise.
///
/// The check a folder has to pass before a project is bound to it: under the
/// App Sandbox a folder the user never picked resolves but can't be read, and
/// binding it would only fail later, on every chat turn.
pub fn reachable_dir(path: &str) -> Option<String> {
    let path = std::fs::canonicalize(expand_home(path)).ok()?;
    std::fs::read_dir(&path).ok()?;
    Some(path.to_string_lossy().into_owned())
}

/// [`reachable_dir`], short of `/`, which a project is never bound to.  The
/// home folder is fine: the indexer and explorer skip hidden folders, so
/// `~/.smartloop` and the like stay out of reach.
pub fn bindable_dir(path: &str) -> Option<String> {
    let dir = reachable_dir(path)?;
    Path::new(&dir).parent().is_some().then_some(dir)
}

/// Whether `dir` is the SLP home (`SLP_HOME`, else `~/.smartloop`) or a
/// folder inside it: the framework's own state, never a folder to explore.
pub fn in_slp_home(dir: &str) -> bool {
    dir_within(dir, &crate::auth::credential_store::slp_home().to_string_lossy()).is_some()
}

/// `path`, canonicalized, when it is `root` or a folder inside it; `None`
/// when it lies elsewhere or either side doesn't resolve.  Canonicalizing
/// both sides first means `..` and symlinks can't step out of `root`.
pub fn dir_within(path: &str, root: &str) -> Option<String> {
    let root = std::fs::canonicalize(expand_home(root)).ok()?;
    let path = std::fs::canonicalize(expand_home(path)).ok()?;
    (path.is_dir() && path.starts_with(&root)).then(|| path.to_string_lossy().into_owned())
}
