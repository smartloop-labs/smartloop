//! Port of `tests/unit/test_download_registry.py`.
//!
//! The registry that lets a second request find a running download.

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use serial_test::serial;
use smartloop::model::downloader::cancellation::{self, CancelToken, DownloadCancelled, DownloadInProgress};

/// The `_clean_registry` autouse fixture.
struct CleanRegistry;

impl CleanRegistry {
    fn new() -> Self {
        cancellation::reset_for_tests();
        CleanRegistry
    }
}

impl Drop for CleanRegistry {
    fn drop(&mut self) {
        cancellation::reset_for_tests();
    }
}

/// `threading.Timer(delay, f).start()`.
fn timer(delay: Duration, f: impl FnOnce() + Send + 'static) {
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        f();
    });
}

// ---------------------------------------------------------------- TestTheToken

#[test]
fn test_check_is_silent_until_cancelled() {
    let token = CancelToken::new();
    token.check("download").unwrap(); // no raise
    token.cancel();
    let err: DownloadCancelled = token.check("download").unwrap_err();
    assert_eq!(err.to_string(), "download cancelled");
}

#[test]
fn test_the_message_names_what_was_stopped() {
    let token = CancelToken::new();
    token.cancel();
    let err = token.check("model.gguf").unwrap_err();
    assert!(err.to_string().contains("model.gguf"));
}

#[test]
fn test_cancelling_twice_is_harmless() {
    let token = CancelToken::new();
    token.cancel();
    token.cancel();
    assert!(token.cancelled());
}

#[test]
fn test_wait_returns_false_when_the_time_runs_out() {
    assert!(!CancelToken::new().wait(0.01));
}

#[test]
fn test_wait_returns_early_when_cancelled() {
    let token = CancelToken::new();
    let remote = token.clone();
    timer(Duration::from_millis(20), move || remote.cancel());
    let started = Instant::now();
    assert!(token.wait(5.0));
    // The point of the substitution: a cancel does not wait out the backoff.
    assert!(started.elapsed() < Duration::from_secs(1));
}

// ----------------------------------------------------------- TestClaimingAModel

#[test]
#[serial]
fn test_begin_hands_out_a_token_and_records_it() {
    let _clean = CleanRegistry::new();
    let token = cancellation::begin("org/repo").unwrap();
    assert!(cancellation::find("org/repo").unwrap().ptr_eq(&token));
    assert_eq!(cancellation::active_keys(), vec!["org/repo".to_string()]);
}

#[test]
#[serial]
fn test_a_second_download_of_the_same_model_is_refused() {
    let _clean = CleanRegistry::new();
    cancellation::begin("org/repo").unwrap();
    let err: DownloadInProgress = cancellation::begin_with_wait("org/repo", 0.0).unwrap_err();
    assert!(err.to_string().contains("org/repo"));
}

#[test]
#[serial]
fn test_a_different_model_is_not_refused() {
    let _clean = CleanRegistry::new();
    cancellation::begin("org/one").unwrap();
    cancellation::begin("org/two").unwrap();
    assert_eq!(cancellation::active_keys(), vec!["org/one".to_string(), "org/two".to_string()]);
}

/// Cancel-then-immediately-re-enable must not read as a duplicate.
#[test]
#[serial]
fn test_begin_waits_out_a_departing_entry() {
    let _clean = CleanRegistry::new();
    let first = cancellation::begin("org/repo").unwrap();
    let departing = first.clone();
    timer(Duration::from_millis(50), move || cancellation::end("org/repo", &departing));

    let second = cancellation::begin_with_wait("org/repo", 5.0).unwrap();
    assert!(!second.ptr_eq(&first));
    assert!(cancellation::find("org/repo").unwrap().ptr_eq(&second));
}

#[test]
#[serial]
fn test_the_refusal_carries_a_409() {
    let _clean = CleanRegistry::new();
    cancellation::begin("org/repo").unwrap();
    let err = cancellation::begin_with_wait("org/repo", 0.0).unwrap_err();
    assert_eq!(err.status_code(), 409);
}

// ---------------------------------------------------------------- TestReleasing

#[test]
#[serial]
fn test_end_releases_the_key() {
    let _clean = CleanRegistry::new();
    let token = cancellation::begin("org/repo").unwrap();
    cancellation::end("org/repo", &token);
    assert!(cancellation::find("org/repo").is_none());
    assert!(cancellation::active_keys().is_empty());
}

#[test]
#[serial]
fn test_end_sets_done() {
    let _clean = CleanRegistry::new();
    let token = cancellation::begin("org/repo").unwrap();
    assert!(!token.done.is_set());
    cancellation::end("org/repo", &token);
    assert!(token.done.is_set());
}

/// A slow wind-down must not unregister the download that replaced it.
///
/// A cancelled transfer can still be inside a 60s socket read while its
/// replacement has already claimed the key; an unconditional pop there would
/// leave the live download with no token for the next cancel.
#[test]
#[serial]
fn test_end_only_pops_its_own_token() {
    let _clean = CleanRegistry::new();
    let stale = cancellation::begin("org/repo").unwrap();
    cancellation::end("org/repo", &stale);
    let live = cancellation::begin("org/repo").unwrap();

    cancellation::end("org/repo", &stale); // the late release lands

    assert!(cancellation::find("org/repo").unwrap().ptr_eq(&live));
}

#[test]
#[serial]
fn test_a_late_release_still_marks_its_own_token_done() {
    let _clean = CleanRegistry::new();
    let stale = cancellation::begin("org/repo").unwrap();
    cancellation::end("org/repo", &stale);
    cancellation::begin("org/repo").unwrap();
    cancellation::end("org/repo", &stale);
    assert!(stale.done.is_set());
}

// ------------------------------------------------------- TestRequestingACancel

#[test]
#[serial]
fn test_it_sets_the_flag_and_returns_the_token() {
    let _clean = CleanRegistry::new();
    let token = cancellation::begin("org/repo").unwrap();
    assert!(cancellation::request_cancel("org/repo").unwrap().ptr_eq(&token));
    assert!(token.cancelled());
}

#[test]
#[serial]
fn test_nothing_running_answers_none() {
    let _clean = CleanRegistry::new();
    assert!(cancellation::request_cancel("org/repo").is_none());
}

#[test]
#[serial]
fn test_a_second_cancel_is_idempotent() {
    let _clean = CleanRegistry::new();
    cancellation::begin("org/repo").unwrap();
    let first = cancellation::request_cancel("org/repo").unwrap();
    let second = cancellation::request_cancel("org/repo").unwrap();
    assert!(first.ptr_eq(&second));
    assert!(first.cancelled());
}

// ------------------------------------------------------------- TestConcurrency

#[test]
#[serial]
fn test_only_one_of_many_racing_claims_wins() {
    let _clean = CleanRegistry::new();
    let winners = Arc::new(parking_lot::Mutex::new(Vec::<CancelToken>::new()));
    let barrier = Arc::new(Barrier::new(8));

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let winners = Arc::clone(&winners);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                if let Ok(token) = cancellation::begin_with_wait("org/repo", 0.0) {
                    winners.lock().push(token);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }

    let winners = winners.lock();
    assert_eq!(winners.len(), 1);
    assert!(cancellation::find("org/repo").unwrap().ptr_eq(&winners[0]));
}
