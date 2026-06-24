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
    """Return a Fernet instance, generating the key file if needed."""
    from cryptography.fernet import Fernet

    kp = _key_path()
    if os.path.exists(kp):
        with open(kp, "rb") as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        _ensure_dir(os.path.dirname(kp))
        fd = os.open(kp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
    return Fernet(key)


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
    fernet = _get_fernet()
    cp = _creds_path()
    _ensure_dir(os.path.dirname(cp))
    fd = os.open(cp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(fernet.encrypt(json.dumps(creds).encode()))


# -- helpers ---------------------------------------------------------------

def _get(key: str) -> str | None:
    return _load_creds().get(key)


def _set(key: str, value: str) -> None:
    creds = _load_creds()
    creds[key] = value
    _save_creds(creds)


def _clear(key: str) -> None:
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


# -- unified resolver ------------------------------------------------------

def get_active_token() -> str | None:
    """Return the best available auth token: access_token > developer_token."""
    return _get(_ACCESS_TOKEN) or _get(_DEVELOPER_TOKEN)


def has_credentials() -> bool:
    """Return True if any auth token is available."""
    return bool(auth_headers())


def auth_headers() -> dict[str, str]:
    """Return the correct HTTP auth headers based on available credentials.

    - access_token  → ``{"Authorization": "Bearer <token>"}``
    - developer_token → ``{"X-Api-Key": "<token>"}``
    - neither → ``{}``
    """
    access = _get(_ACCESS_TOKEN)
    if access:
        return {"Authorization": f"Bearer {access}"}
    dev = _get(_DEVELOPER_TOKEN)
    if dev:
        return {"X-Api-Key": f"{dev}"}
    return {}
