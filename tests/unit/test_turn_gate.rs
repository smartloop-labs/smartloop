//! Indexing waits while a turn is answered, and goes on once it's done.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serial_test::serial;
use smartloop::tasks::turn_gate::{begin_turn, turn_active, wait_for_turns, wait_for_turns_at_most};

#[test]
#[serial]
fn with_no_turn_indexing_goes_on_at_once() {
    assert!(!turn_active());
    assert!(!wait_for_turns(), "nothing to wait for");
}

#[test]
#[serial]
fn indexing_waits_for_the_turn_and_resumes_when_it_ends() {
    let turn = begin_turn();
    assert!(turn_active());
    let resumed = Arc::new(AtomicBool::new(false));
    let indexer = {
        let resumed = resumed.clone();
        std::thread::spawn(move || {
            let waited = wait_for_turns();
            resumed.store(true, Ordering::SeqCst);
            waited
        })
    };
    std::thread::sleep(Duration::from_millis(150));
    assert!(!resumed.load(Ordering::SeqCst), "paused while the turn is answered");
    drop(turn);
    assert!(indexer.join().unwrap(), "it waited");
    assert!(!turn_active());
}

#[test]
#[serial]
fn overlapping_turns_hold_it_until_the_last_one_ends() {
    let first = begin_turn();
    let second = begin_turn();
    drop(first);
    assert!(turn_active(), "one still answered");
    drop(second);
    assert!(!turn_active());
}

#[test]
#[serial]
fn a_turn_that_never_ends_holds_it_only_so_long() {
    let _stuck = begin_turn();
    let started = Instant::now();
    assert!(wait_for_turns_at_most(Duration::from_millis(100)));
    let waited = started.elapsed();
    assert!(waited >= Duration::from_millis(100) && waited < Duration::from_secs(2), "{waited:?}");
}
