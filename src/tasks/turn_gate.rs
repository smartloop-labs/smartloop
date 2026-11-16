//! Indexing waits while a chat turn is answered.
//!
//! Embedding and generation share one device: an indexing pass running
//! through a reply slows it to a crawl, and on an integrated GPU it can be
//! most of the wait. A turn takes a [`TurnGuard`] once it has waited for the
//! folder's index (see `INDEX_WAIT` in the chat router) and holds it until
//! its stream ends; indexers call [`wait_for_turns`] before each file, so a
//! pass already under way pauses between files and picks up after.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

static ACTIVE: AtomicUsize = AtomicUsize::new(0);
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
    let deadline = Instant::now() + cap;
    let mut lock = LOCK.lock();
    while turn_active() && Instant::now() < deadline {
        DONE.wait_until(&mut lock, deadline);
    }
    true
}
