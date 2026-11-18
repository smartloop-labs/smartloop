//! Indexing waits while a chat turn is answered.
//!
//! Embedding and generation share one device: an indexing pass running
//! through a reply slows it to a crawl, and on an integrated GPU it can be
//! most of the wait. A turn takes a [`TurnGuard`] once it has waited for the
//! folder's index (see `INDEX_WAIT` in the chat router) and holds it until
//! its stream ends; indexers call [`wait_for_turns`] before each file, so a
//! pass already under way pauses between files and picks up after.
//!
//! Under `SLP_TIMING` each pause is logged as `index.paused`, and each file
//! indexed while a turn was answered as `index.overlap`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use serde_json::json;

use crate::utils::timing;

static ACTIVE: AtomicUsize = AtomicUsize::new(0);
/// Turns begun so far, to tell whether one began during a stretch of work.
static STARTED: AtomicUsize = AtomicUsize::new(0);
static LOCK: Mutex<()> = Mutex::new(());
static DONE: Condvar = Condvar::new();

/// The longest one wait holds an indexer back: a turn that never ends (a
/// stream nobody reads) mustn't stall the index for good.
pub const MAX_PAUSE: Duration = Duration::from_secs(300);

/// A turn being answered; indexing waits while any is held.
#[must_use = "indexing resumes as soon as the guard is dropped"]
pub struct TurnGuard {
    _private: (),
}

impl Drop for TurnGuard {
    fn drop(&mut self) {
        let _lock = LOCK.lock();
        ACTIVE.fetch_sub(1, Ordering::SeqCst);
        DONE.notify_all();
    }
}

/// Mark a turn as being answered until the guard is dropped.
pub fn begin_turn() -> TurnGuard {
    ACTIVE.fetch_add(1, Ordering::SeqCst);
    STARTED.fetch_add(1, Ordering::SeqCst);
    TurnGuard { _private: () }
}

/// Whether a turn is being answered.
pub fn turn_active() -> bool {
    ACTIVE.load(Ordering::SeqCst) > 0
}

/// Wait until no turn is being answered, for at most [`MAX_PAUSE`]. Whether
/// it waited at all.
pub fn wait_for_turns() -> bool {
    wait_for_turns_at_most(MAX_PAUSE)
}

/// [`wait_for_turns`] with its own cap.
pub fn wait_for_turns_at_most(cap: Duration) -> bool {
    if !turn_active() {
        return false;
    }
    let started = Instant::now();
    let deadline = started + cap;
    let mut lock = LOCK.lock();
    while turn_active() && Instant::now() < deadline {
        DONE.wait_until(&mut lock, deadline);
    }
    drop(lock);
    timing::event("index.paused", json!({"ms": timing::ms(started.elapsed()), "capped": turn_active()}));
    true
}

/// One file of indexing, timed under `SLP_TIMING`: the gate pauses only
/// between files, so a file under way when a turn begins shares the device
/// with it, and is logged as `index.overlap`.
pub struct IndexWork {
    source: &'static str,
    file: String,
    started: Instant,
    turn_active: bool,
    turns: usize,
}

/// Start timing one file of indexing from `source`; `None` with timing off.
pub fn index_work(source: &'static str, file: &str) -> Option<IndexWork> {
    timing::on().then(|| IndexWork {
        source,
        file: file.to_string(),
        started: Instant::now(),
        turn_active: turn_active(),
        turns: STARTED.load(Ordering::SeqCst),
    })
}

impl IndexWork {
    /// Whether a turn was answered at any point while this file was indexed.
    pub fn overlapped(&self) -> bool {
        self.turn_active || STARTED.load(Ordering::SeqCst) != self.turns
    }
}

impl Drop for IndexWork {
    fn drop(&mut self) {
        if self.overlapped() {
            timing::event(
                "index.overlap",
                json!({"source": self.source, "file": self.file, "ms": timing::ms(self.started.elapsed())}),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_overlaps_any_turn_answered_while_it_ran() {
        timing::testing::capture();
        let quiet = index_work("workspace", "a.md").unwrap();
        assert!(!quiet.overlapped());
        let brief = index_work("workspace", "b.md").unwrap();
        drop(begin_turn());
        assert!(brief.overlapped(), "a turn began and ended while it ran");
        let turn = begin_turn();
        let during = index_work("documents", "c.pdf").unwrap();
        drop(turn);
        assert!(during.overlapped(), "begun while a turn was answered");
    }
}
