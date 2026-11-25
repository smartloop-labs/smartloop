//! How many connections a download uses, and how they are shared between
//! its files.
//!
//! Kept apart from the HTTP transport: nothing here assumes blocking reqwest
//! or one connection per range.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

/// Fewest connections a download is tuned down to.
pub const MIN_CONNECTIONS: usize = 2;
/// Most connections a download is tuned up to.  Bounded by the blocking
/// transport, which drives every stream from one internal runtime.
pub const MAX_CONNECTIONS: usize = 16;

/// Starting connection count by bytes still to fetch: `(at least, count)`,
/// largest first.  A small download finishes before tuning has much to say,
/// and a large one runs long enough that starting low would waste its first
/// minute ramping up.
pub const STARTING_CONNECTIONS: &[(u64, usize)] = &[
    (2 * 1024 * 1024 * 1024, 8),
    (100 * 1024 * 1024, 4),
    (0, 1),
];

/// The connection count to start a download of `remaining` bytes with.
pub fn starting_connections(remaining: u64) -> usize {
    STARTING_CONNECTIONS
        .iter()
        .find(|(at_least, _)| remaining >= *at_least)
        .map_or(1, |(_, count)| *count)
        .min(MAX_CONNECTIONS)
}

/// How often a waiter re-checks whether it should give up.
const RECHECK: Duration = Duration::from_millis(50);

#[derive(Debug)]
struct PermitState {
    limit: usize,
    in_use: usize,
}

/// The connections one download may have open at once, shared by all of its
/// files.  The limit can move while the download runs; lowering it closes
/// nothing, it just holds back new requests until enough have finished.
#[derive(Debug)]
pub struct ConnectionPermits {
    state: Mutex<PermitState>,
    freed: Condvar,
}

impl ConnectionPermits {
    /// A pool allowing `limit` connections (at least one).
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self { state: Mutex::new(PermitState { limit: limit.max(1), in_use: 0 }), freed: Condvar::new() })
    }

    /// The current limit.
    pub fn limit(&self) -> usize {
        self.state.lock().limit
    }

    /// Connections open right now.
    pub fn in_use(&self) -> usize {
        self.state.lock().in_use
    }

    /// Change the limit (at least one).
    pub fn set_limit(&self, limit: usize) {
        self.state.lock().limit = limit.max(1);
        self.freed.notify_all();
    }

    /// Wait for a free connection.  `None` once `give_up` says there is no
    /// longer anything to use it for.
    pub fn acquire(self: &Arc<Self>, give_up: impl Fn() -> bool) -> Option<Permit> {
        let mut state = self.state.lock();
        loop {
            if give_up() {
                return None;
            }
            if state.in_use < state.limit {
                state.in_use += 1;
                return Some(Permit { pool: Arc::clone(self) });
            }
            self.freed.wait_for(&mut state, RECHECK);
        }
    }
}

/// One open connection's claim on a [`ConnectionPermits`]; released on drop.
#[derive(Debug)]
pub struct Permit {
    pool: Arc<ConnectionPermits>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.pool.state.lock().in_use -= 1;
        self.pool.freed.notify_all();
    }
}
