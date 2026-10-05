//! Port of `smartloop/platform/auth.py`.
//!
//! Authentication bridge for Smartloop platform calls.
//!
//! Currently delegates to `credential_store`. Will later be swapped to call
//! the local server's /auth endpoints when clients become stateless.

use std::collections::BTreeMap;

use crate::auth::credential_store;

/// Return HTTP auth headers for Smartloop API calls.
pub fn auth_headers() -> BTreeMap<String, String> {
    credential_store::auth_headers()
}

/// Return true if valid credentials are available.
pub fn has_credentials() -> bool {
    credential_store::has_credentials()
}
