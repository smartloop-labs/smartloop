"""Credential storage for Smartloop authentication.

Uses a local Fernet-encrypted file store at ``SLP_HOME/.credentials``
with the encryption key kept in ``SLP_HOME/.key``.

Resolution order for developer tokens:
  1. ``-t`` flag (stored via ``set_token`` before any API calls)
  2. File-based credential store
  3. ``SLP_DEVELOPER_TOKEN`` environment variable
"""

import json
import logging
import os
import stat
import threading

logger = logging.getLogger(__name__)

_DEVELOPER_TOKEN = "developer_token"
_ACCESS_TOKEN = "access_token"
_REFRESH_TOKEN = "refresh_token"


# -- file-based storage ----------------------------------------------------

def _slp_home() -> str:
    return os.path.expanduser(os.environ.get("SLP_HOME", "~/.smartloop"))


def _key_path() -> str:
    return os.path.join(_slp_home(), ".key")


def _creds_path() -> str:
    return os.path.join(_slp_home(), ".credentials")


def _ensure_dir(path: str) -> None:
    """Create the directory if needed and lock permissions to owner-only."""
    os.makedirs(path, exist_ok=True)
    os.chmod(path, stat.S_IRWXU)  # 700


def _get_fernet():
    """Return a Fernet instance, generating the key file if needed.

    The key is created with ``O_EXCL`` and never truncated in place.  Truncating
    was the same hazard :func:`_save_creds` was hardened against, and worse here:
    this runs from :func:`_load_creds`, which readers reach *without* the write
    lock, so a reader arriving mid-write got a short key, ``Fernet`` raised, and
    the bare ``except`` below reported it as "no credentials" — a phantom logout.

    Losing the ``O_EXCL`` race means another writer created the key first; read
    theirs rather than overwriting it, which would strand every credential
    already encrypted under it.
    """
    from cryptography.fernet import Fernet

    kp = _key_path()
    if not os.path.exists(kp):
        _ensure_dir(os.path.dirname(kp))
        try:
            fd = os.open(kp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as f:
                f.write(Fernet.generate_key())
                f.flush()
                os.fsync(f.fileno())

    with open(kp, "rb") as f:
        return Fernet(f.read())


def _load_creds() -> dict:
    cp = _creds_path()
    if not os.path.exists(cp):
        return {}
    try:
        fernet = _get_fernet()
        with open(cp, "rb") as f:
            return json.loads(fernet.decrypt(f.read()))
    except Exception:
        return {}


def _save_creds(creds: dict) -> None:
    """Write the credential file, replacing it in one step.

    Writing in place truncated the file before the ciphertext landed, so a
    reader arriving mid-write — or a process that died there — found a partial
    Fernet blob.  ``_load_creds`` cannot tell that apart from a bad key and
    answers ``{}`` either way, which presents as a silent, total logout.  Write
    a sibling and rename: readers see the whole of one version or the whole of
    the other.
    """
    fernet = _get_fernet()
    cp = _creds_path()
    _ensure_dir(os.path.dirname(cp))
    tmp = f"{cp}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(fernet.encrypt(json.dumps(creds).encode()))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cp)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# -- helpers ---------------------------------------------------------------

#: Held across the read-modify-write in :func:`_set` and :func:`_clear`.  The
#: whole file is rewritten to change one key, so two unsynchronized writers —
#: the token refresh agent storing an access token while a request thread stores
#: the rotated refresh token — would each save a copy of the file as it looked
#: before the other's change, dropping one of the two silently.
#:
#: Process-local.  A second process (the CLI alongside a running server) can
#: still interleave; :func:`_save_creds` keeps that from corrupting the file,
#: but the lost-update window remains and wants a real file lock.
#:
#: The supervisor's agent children are deliberately not in that set.  They read
#: this store constantly and never write it: the one write they could reach,
#: redeeming the refresh token, is declined in a pinned process
#: (:func:`smartloop.platform.token_agent.refresh_access_token_now`).  Readers
#: take no lock at all and do not need one — :func:`_save_creds` renames a
#: finished file into place, so a reader sees one whole version or the other.
_write_lock = threading.RLock()


def _get(key: str) -> str | None:
    return _load_creds().get(key)


def _set(key: str, value: str) -> None:
    with _write_lock:
        creds = _load_creds()
        creds[key] = value
        _save_creds(creds)


def _clear(key: str) -> None:
    with _write_lock:
        creds = _load_creds()
        creds.pop(key, None)
        _save_creds(creds)


# -- developer token -------------------------------------------------------

def get_token() -> str | None:
    """Retrieve the developer token: file store first, then env var."""
    return _get(_DEVELOPER_TOKEN) or os.environ.get("SLP_DEVELOPER_TOKEN")


def set_token(token: str) -> None:
    """Store a developer token."""
    _set(_DEVELOPER_TOKEN, token)


def clear_token() -> None:
    """Remove the stored developer token."""
    _clear(_DEVELOPER_TOKEN)


# -- JWT access token (set by the Electron client after OAuth) -------------

def get_access_token() -> str | None:
    return _get(_ACCESS_TOKEN)


def set_access_token(token: str) -> None:
    _set(_ACCESS_TOKEN, token)


def clear_access_token() -> None:
    _clear(_ACCESS_TOKEN)


# -- refresh token ---------------------------------------------------------

def get_refresh_token() -> str | None:
    return _get(_REFRESH_TOKEN)


def set_refresh_token(token: str) -> None:
    _set(_REFRESH_TOKEN, token)


def clear_refresh_token() -> None:
    _clear(_REFRESH_TOKEN)


# -- the OAuth pair, written as one --------------------------------------

def set_tokens(access: str, refresh: str | None = None) -> None:
    """Store the access token, and the rotated refresh token with it.

    One file version for the pair.  Storing them with two calls leaves a
    moment where the new access token is on disk beside the refresh token that
    was just spent to mint it, and a crash or a kill in that moment makes the
    state permanent: the next redemption presents a spent token, is refused,
    and reads the refusal as "re-login required".  A single-use refresh token
    makes that window unrecoverable rather than merely untidy.

    *refresh* is optional because the endpoint does not always rotate; ``None``
    leaves the stored one alone.
    """
    with _write_lock:
        creds = _load_creds()
        creds[_ACCESS_TOKEN] = access
        if refresh:
            creds[_REFRESH_TOKEN] = refresh
        _save_creds(creds)


def clear_tokens() -> None:
    """Drop both OAuth credentials in one file version.

    The counterpart to :func:`set_tokens`, for the refusal that ends the
    session: clearing them one at a time leaves a readable state holding a
    refresh token whose access token is already gone.
    """
    with _write_lock:
        creds = _load_creds()
        creds.pop(_ACCESS_TOKEN, None)
        creds.pop(_REFRESH_TOKEN, None)
        _save_creds(creds)


# -- unified resolver ------------------------------------------------------

def get_active_token() -> str | None:
    """Return the best available auth token: access_token > developer_token."""
    return _get(_ACCESS_TOKEN) or get_token()


def has_credentials() -> bool:
    """Return True if any auth token is available."""
    return bool(auth_headers())


def auth_headers() -> dict[str, str]:
    """Return the correct HTTP auth headers based on available credentials.

    - access_token  → ``{"Authorization": "Bearer <token>"}``
    - developer_token → ``{"X-Api-Key": "<token>"}``
    - neither → ``{}``

    The developer token comes from :func:`get_token`, so ``SLP_DEVELOPER_TOKEN``
    counts here exactly as the module docstring says it does.  Reading the file
    store directly meant an env-var-only user was reported as signed in by
    ``/v1/auth/status`` — which asks :func:`get_token` — while every request
    went out with no credentials at all: web search silently off, the catalog
    silently anonymous.
    """
    access = _get(_ACCESS_TOKEN)
    if access:
        return {"Authorization": f"Bearer {access}"}
    dev = get_token()
    if dev:
        return {"X-Api-Key": f"{dev}"}
    return {}
