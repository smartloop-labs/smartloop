//! Port of `tests/unit/test_credential_store.py`.
//!
//! Every test runs inside [`isolate_credential_store`] (the conftest autouse
//! fixture), so the store lives in a temp directory.

use std::collections::BTreeMap;

use serial_test::serial;
use smartloop::auth::credential_store;

use crate::shared::core_support::isolate_credential_store;

fn headers(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

mod test_developer_token {
    use super::*;

    #[test]
    #[serial]
    fn test_set_and_get() {
        let _g = isolate_credential_store();
        credential_store::set_token("sk-test-123");
        assert_eq!(credential_store::get_token().as_deref(), Some("sk-test-123"));
    }

    #[test]
    #[serial]
    fn test_clear() {
        let _g = isolate_credential_store();
        credential_store::set_token("sk-test-123");
        credential_store::clear_token();
        assert_eq!(credential_store::get_token(), None);
    }

    #[test]
    #[serial]
    fn test_env_var_fallback() {
        let g = isolate_credential_store();
        g.set_developer_token_env("sk-from-env");
        assert_eq!(credential_store::get_token().as_deref(), Some("sk-from-env"));
    }

    #[test]
    #[serial]
    fn test_keychain_takes_precedence_over_env() {
        let g = isolate_credential_store();
        g.set_developer_token_env("sk-from-env");
        credential_store::set_token("sk-from-store");
        assert_eq!(credential_store::get_token().as_deref(), Some("sk-from-store"));
    }
}

mod test_access_token {
    use super::*;

    #[test]
    #[serial]
    fn test_set_and_get() {
        let _g = isolate_credential_store();
        credential_store::set_access_token("jwt-abc");
        assert_eq!(credential_store::get_access_token().as_deref(), Some("jwt-abc"));
    }

    #[test]
    #[serial]
    fn test_clear() {
        let _g = isolate_credential_store();
        credential_store::set_access_token("jwt-abc");
        credential_store::clear_access_token();
        assert_eq!(credential_store::get_access_token(), None);
    }
}

mod test_refresh_token {
    use super::*;

    #[test]
    #[serial]
    fn test_set_and_get() {
        let _g = isolate_credential_store();
        credential_store::set_refresh_token("rt-xyz");
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt-xyz"));
    }

    #[test]
    #[serial]
    fn test_clear() {
        let _g = isolate_credential_store();
        credential_store::set_refresh_token("rt-xyz");
        credential_store::clear_refresh_token();
        assert_eq!(credential_store::get_refresh_token(), None);
    }
}

mod test_active_token {
    use super::*;

    #[test]
    #[serial]
    fn test_prefers_access_over_developer() {
        let _g = isolate_credential_store();
        credential_store::set_token("sk-dev");
        credential_store::set_access_token("jwt-abc");
        assert_eq!(credential_store::get_active_token().as_deref(), Some("jwt-abc"));
    }

    #[test]
    #[serial]
    fn test_falls_back_to_developer() {
        let _g = isolate_credential_store();
        credential_store::set_token("sk-dev");
        assert_eq!(credential_store::get_active_token().as_deref(), Some("sk-dev"));
    }

    #[test]
    #[serial]
    fn test_none_when_empty() {
        let _g = isolate_credential_store();
        assert_eq!(credential_store::get_active_token(), None);
    }
}

mod test_has_credentials {
    use super::*;

    #[test]
    #[serial]
    fn test_false_when_empty() {
        let _g = isolate_credential_store();
        assert!(!credential_store::has_credentials());
    }

    #[test]
    #[serial]
    fn test_true_with_developer_token() {
        let _g = isolate_credential_store();
        credential_store::set_token("sk-dev");
        assert!(credential_store::has_credentials());
    }

    #[test]
    #[serial]
    fn test_true_with_access_token() {
        let _g = isolate_credential_store();
        credential_store::set_access_token("jwt-abc");
        assert!(credential_store::has_credentials());
    }
}

mod test_auth_headers {
    use super::*;

    #[test]
    #[serial]
    fn test_bearer_when_access_token() {
        let _g = isolate_credential_store();
        credential_store::set_access_token("jwt-abc");
        assert_eq!(credential_store::auth_headers(), headers(&[("Authorization", "Bearer jwt-abc")]));
    }

    #[test]
    #[serial]
    fn test_api_key_when_developer_token() {
        let _g = isolate_credential_store();
        credential_store::set_token("sk-dev");
        assert_eq!(credential_store::auth_headers(), headers(&[("X-Api-Key", "sk-dev")]));
    }

    #[test]
    #[serial]
    fn test_bearer_preferred_over_api_key() {
        let _g = isolate_credential_store();
        credential_store::set_token("sk-dev");
        credential_store::set_access_token("jwt-abc");
        assert_eq!(credential_store::auth_headers(), headers(&[("Authorization", "Bearer jwt-abc")]));
    }

    #[test]
    #[serial]
    fn test_empty_when_no_tokens() {
        let _g = isolate_credential_store();
        assert_eq!(credential_store::auth_headers(), headers(&[]));
    }
}

mod test_file_backend_persistence {
    use super::*;

    #[test]
    #[serial]
    fn test_credentials_file_created() {
        let g = isolate_credential_store();
        credential_store::set_token("sk-test");
        assert!(g.path().join(".credentials").exists());
        assert!(g.path().join(".key").exists());
    }

    #[cfg(unix)]
    #[test]
    #[serial]
    fn test_credentials_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let g = isolate_credential_store();
        credential_store::set_token("sk-test");
        let key_stat = std::fs::metadata(g.path().join(".key")).unwrap();
        assert_eq!(key_stat.permissions().mode() & 0o777, 0o600);
    }
}

/// `SLP_DEVELOPER_TOKEN` is step 3 of the documented resolution order.
///
/// Only `get_token` honoured it.  `auth_headers` and `get_active_token` read
/// the file store directly, so an env-var-only user was reported as signed in
/// by `/v1/auth/status` — which asks `get_token` — while every outbound
/// request carried no credentials at all, and the web-search gate (which asks
/// `has_credentials`) turned itself off.
mod test_the_env_var_is_a_real_credential {
    use super::*;

    #[test]
    #[serial]
    fn test_it_produces_an_api_key_header() {
        let g = isolate_credential_store();
        g.set_developer_token_env("sk-env");
        assert_eq!(credential_store::auth_headers(), headers(&[("X-Api-Key", "sk-env")]));
        assert!(credential_store::has_credentials());
    }

    #[test]
    #[serial]
    fn test_it_is_the_active_token() {
        let g = isolate_credential_store();
        g.set_developer_token_env("sk-env");
        assert_eq!(credential_store::get_active_token().as_deref(), Some("sk-env"));
    }

    #[test]
    #[serial]
    fn test_a_stored_token_still_wins() {
        let g = isolate_credential_store();
        g.set_developer_token_env("sk-env");
        credential_store::set_token("sk-stored");
        assert_eq!(credential_store::auth_headers(), headers(&[("X-Api-Key", "sk-stored")]));
    }

    #[test]
    #[serial]
    fn test_an_access_token_still_wins() {
        let g = isolate_credential_store();
        g.set_developer_token_env("sk-env");
        credential_store::set_access_token("jwt-abc");
        assert_eq!(credential_store::auth_headers(), headers(&[("Authorization", "Bearer jwt-abc")]));
    }
}

/// Changing one key rewrites the whole file.
///
/// A refresh stores an access token and a rotated refresh token back to back,
/// and the background loop can be doing the same thing on another thread.
/// Unsynchronized, each writer saved the file as it looked before the other's
/// change — so one of the two tokens vanished, and a half-written Fernet blob
/// reads as no credentials at all.
mod test_concurrent_writes_keep_both {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    #[serial]
    fn test_keys_written_at_once_all_survive() {
        let _g = isolate_credential_store();
        let writers: Vec<(fn(&str), &'static str)> = vec![
            (credential_store::set_access_token, "jwt-abc"),
            (credential_store::set_refresh_token, "rt-xyz"),
            (credential_store::set_token, "sk-dev"),
        ];
        let barrier = Arc::new(Barrier::new(writers.len()));

        let handles: Vec<_> = writers
            .into_iter()
            .map(|(f, value)| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..25 {
                        f(value);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(credential_store::get_access_token().as_deref(), Some("jwt-abc"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("rt-xyz"));
        assert_eq!(credential_store::get_token().as_deref(), Some("sk-dev"));
    }

    #[test]
    #[serial]
    fn test_a_reader_never_sees_a_half_written_file() {
        let _g = isolate_credential_store();
        credential_store::set_access_token("jwt-original");
        let stop = Arc::new(AtomicBool::new(false));

        let reader = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut seen = Vec::new();
                while !stop.load(Ordering::SeqCst) {
                    seen.push(credential_store::get_access_token());
                }
                seen
            })
        };
        for i in 0..50 {
            credential_store::set_access_token(&format!("jwt-{i}"));
        }
        stop.store(true, Ordering::SeqCst);
        let seen = reader.join().unwrap();

        // Every read landed on one whole version or another — never on
        // `None`, which is what a torn file decrypts to.
        assert!(!seen.is_empty() && seen.iter().all(Option::is_some));
    }
}

/// A rotated pair reaches disk in one file version, or not at all.
///
/// Storing the two halves separately leaves a readable state holding the new
/// access token beside the refresh token that was just spent to mint it.  A
/// crash there makes it permanent: the next redemption presents a spent
/// token, is refused, and reads the refusal as "re-login required".  With a
/// single-use refresh token that window is unrecoverable rather than untidy.
mod test_the_oauth_pair_is_written_as_one {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use serde_json::Value;

    use super::*;

    /// Record a copy of every `save_creds` argument (the Python tests wrap
    /// `_save_creds` with a recording lambda).
    fn record_writes() -> Arc<Mutex<Vec<credential_store::Creds>>> {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&writes);
        credential_store::set_save_observer(Some(Box::new(move |creds| sink.lock().push(creds.clone()))));
        writes
    }

    fn field<'a>(creds: &'a credential_store::Creds, key: &str) -> Option<&'a str> {
        creds.get(key).and_then(Value::as_str)
    }

    #[test]
    #[serial]
    fn test_a_rotated_pair_is_one_write() {
        let _g = isolate_credential_store();
        let writes = record_writes();

        credential_store::set_tokens("new-at", Some("new-rt"));

        let writes = writes.lock().clone();
        assert_eq!(writes.len(), 1, "the pair reached disk in {} versions", writes.len());
        assert_eq!(field(&writes[0], "access_token"), Some("new-at"));
        assert_eq!(field(&writes[0], "refresh_token"), Some("new-rt"));
    }

    #[test]
    #[serial]
    fn test_no_version_pairs_a_new_access_token_with_the_old_refresh_token() {
        let _g = isolate_credential_store();
        credential_store::set_tokens("old-at", Some("old-rt"));

        let seen = record_writes();

        credential_store::set_tokens("new-at", Some("new-rt"));

        let torn: Vec<_> = seen
            .lock()
            .iter()
            .filter(|c| field(c, "access_token") == Some("new-at") && field(c, "refresh_token") == Some("old-rt"))
            .cloned()
            .collect();
        assert!(torn.is_empty(), "a reader could observe the new access token beside the spent refresh token");
    }

    /// The endpoint does not always rotate; `None` must not erase.
    #[test]
    #[serial]
    fn test_an_unrotated_refresh_token_is_left_alone() {
        let _g = isolate_credential_store();
        credential_store::set_tokens("old-at", Some("old-rt"));
        credential_store::set_tokens("new-at", None);

        assert_eq!(credential_store::get_access_token().as_deref(), Some("new-at"));
        assert_eq!(credential_store::get_refresh_token().as_deref(), Some("old-rt"));
    }

    #[test]
    #[serial]
    fn test_clearing_drops_both_in_one_version() {
        let _g = isolate_credential_store();
        credential_store::set_tokens("at", Some("rt"));

        let writes = record_writes();

        credential_store::clear_tokens();

        assert_eq!(writes.lock().len(), 1);
        assert_eq!(credential_store::get_access_token(), None);
        assert_eq!(credential_store::get_refresh_token(), None);
    }
}
