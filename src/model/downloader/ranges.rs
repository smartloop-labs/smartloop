//! How many connections a download uses, and how they are shared between
//! its files.
//!
//! Kept apart from the HTTP transport: nothing here assumes blocking reqwest
//! or one connection per range.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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

/// Time since an arbitrary origin; injectable so tests need not sleep.
pub type Clock = Arc<dyn Fn() -> Duration + Send + Sync>;

/// A [`Clock`] reading real time.
pub fn system_clock() -> Clock {
    let origin = Instant::now();
    Arc::new(move || origin.elapsed())
}

/// How far back a connection's rate looks.  A connection is judged only once
/// it has been open this long, so slow start and a slow first byte don't
/// count against it.
pub const RATE_WINDOW: Duration = Duration::from_secs(30);
/// Below this many bytes per second over [`RATE_WINDOW`] a connection is
/// stalled, whatever the others are doing.
pub const STALL_RATE: f64 = 16.0 * 1024.0;
/// A connection under this fraction of the median of the download's other
/// connections is slow.
pub const SLOW_FRACTION: f64 = 0.25;
/// Other judged connections needed before "slower than the rest" means
/// anything.
pub const MIN_PEERS: usize = 2;

/// Bytes received over a sliding window.
#[derive(Debug)]
struct RateWindow {
    opened: Duration,
    samples: VecDeque<(Duration, u64)>,
    /// Sum of `samples`.
    bytes: u64,
}

impl RateWindow {
    fn new(now: Duration) -> Self {
        Self { opened: now, samples: VecDeque::new(), bytes: 0 }
    }

    fn add(&mut self, now: Duration, bytes: u64) {
        self.samples.push_back((now, bytes));
        self.bytes += bytes;
        self.expire(now);
    }

    fn expire(&mut self, now: Duration) {
        while let Some(&(t, b)) = self.samples.front() {
            if now.saturating_sub(t) <= RATE_WINDOW {
                break;
            }
            self.samples.pop_front();
            self.bytes -= b;
        }
    }

    /// Bytes per second over the window, or since opening when that is
    /// shorter.  `None` in the first second.
    fn rate(&mut self, now: Duration) -> Option<f64> {
        self.expire(now);
        let span = now.saturating_sub(self.opened).min(RATE_WINDOW);
        if span < Duration::from_secs(1) {
            return None;
        }
        Some(self.bytes as f64 / span.as_secs_f64())
    }

    fn mature(&self, now: Duration) -> bool {
        now.saturating_sub(self.opened) >= RATE_WINDOW
    }
}

/// Why a connection should be dropped and its request reissued.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Slowness {
    /// Under [`STALL_RATE`].
    Stalled { rate: f64 },
    /// Under [`SLOW_FRACTION`] of the median of the other connections.
    BehindPeers { rate: f64, median: f64 },
}

impl std::fmt::Display for Slowness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kb = |r: f64| r / 1024.0;
        match self {
            Slowness::Stalled { rate } => write!(f, "stalled at {:.1} KB/s", kb(*rate)),
            Slowness::BehindPeers { rate, median } => {
                write!(f, "at {:.1} KB/s, far behind the other connections' {:.1} KB/s", kb(*rate), kb(*median))
            }
        }
    }
}

/// Throughput of one download's connections, shared by all of its files.
pub struct TransferStats {
    clock: Clock,
    next_id: AtomicU64,
    /// Rates of connections open long enough to be judged.
    judged: Mutex<HashMap<u64, f64>>,
    total: Mutex<RateWindow>,
    received: AtomicU64,
}

impl TransferStats {
    /// Stats read against `clock`.
    pub fn new(clock: Clock) -> Arc<Self> {
        let now = clock();
        Arc::new(Self {
            clock,
            next_id: AtomicU64::new(0),
            judged: Mutex::new(HashMap::new()),
            total: Mutex::new(RateWindow::new(now)),
            received: AtomicU64::new(0),
        })
    }

    /// The clock's reading.
    pub fn now(&self) -> Duration {
        (self.clock)()
    }

    /// Start metering a newly opened connection.
    pub fn open(self: &Arc<Self>) -> ConnectionMeter {
        ConnectionMeter {
            stats: Arc::clone(self),
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            window: RateWindow::new(self.now()),
        }
    }

    /// Bytes received across every connection so far.
    pub fn received(&self) -> u64 {
        self.received.load(Ordering::Relaxed)
    }

    /// Bytes per second across every connection over [`RATE_WINDOW`].
    pub fn total_rate(&self) -> f64 {
        self.total.lock().rate(self.now()).unwrap_or(0.0)
    }

    /// Median rate of the judged connections other than `id`, when there
    /// are at least [`MIN_PEERS`].
    fn median_excluding(&self, id: u64) -> Option<f64> {
        let mut rates: Vec<f64> = self.judged.lock().iter().filter(|(k, _)| **k != id).map(|(_, r)| *r).collect();
        if rates.len() < MIN_PEERS {
            return None;
        }
        rates.sort_by(|a, b| a.total_cmp(b));
        let mid = rates.len() / 2;
        Some(if rates.len() % 2 == 0 { (rates[mid - 1] + rates[mid]) / 2.0 } else { rates[mid] })
    }
}

/// One open connection's view of its own throughput.
pub struct ConnectionMeter {
    stats: Arc<TransferStats>,
    id: u64,
    window: RateWindow,
}

impl ConnectionMeter {
    /// Record `bytes` received now.
    pub fn add(&mut self, bytes: u64) {
        let now = self.stats.now();
        self.window.add(now, bytes);
        self.stats.total.lock().add(now, bytes);
        self.stats.received.fetch_add(bytes, Ordering::Relaxed);
        if self.window.mature(now) {
            if let Some(rate) = self.window.rate(now) {
                self.stats.judged.lock().insert(self.id, rate);
            }
        }
    }

    /// Whether this connection should be dropped and reissued.  Only once it
    /// has been open for a full [`RATE_WINDOW`].
    pub fn slowness(&mut self) -> Option<Slowness> {
        let now = self.stats.now();
        if !self.window.mature(now) {
            return None;
        }
        let rate = self.window.rate(now)?;
        if rate < STALL_RATE {
            return Some(Slowness::Stalled { rate });
        }
        let median = self.stats.median_excluding(self.id)?;
        (rate < median * SLOW_FRACTION).then_some(Slowness::BehindPeers { rate, median })
    }
}

impl Drop for ConnectionMeter {
    fn drop(&mut self) {
        self.stats.judged.lock().remove(&self.id);
    }
}

/// Smallest piece an in-flight range is split into.  Below this, the round
/// trip to open another request costs more than sharing the work saves.
pub const MIN_SPLIT: u64 = 16 * 1024 * 1024;

#[derive(Debug)]
struct CursorState {
    /// First byte not yet claimed by the owner.
    next: u64,
    /// Last byte of the range (inclusive); lowered by a split.
    end: u64,
}

/// One range being fetched: where its owner has got to, and where it ends.
/// The owner claims bytes before writing them and a split only takes bytes
/// beyond the claim, so the two never write the same byte.
#[derive(Debug)]
pub struct RangeCursor {
    start: u64,
    state: Mutex<CursorState>,
}

impl RangeCursor {
    /// A cursor over `start..=end`.
    pub fn new(start: u64, end: u64) -> Arc<Self> {
        Arc::new(Self { start, state: Mutex::new(CursorState { next: start, end }) })
    }

    /// First byte of the range as it was handed out.
    pub fn start(&self) -> u64 {
        self.start
    }

    /// `(next unclaimed byte, last byte)`.
    pub fn position(&self) -> (u64, u64) {
        let state = self.state.lock();
        (state.next, state.end)
    }

    /// Bytes the owner has claimed so far.
    pub fn claimed(&self) -> u64 {
        self.state.lock().next - self.start
    }

    /// Claim up to `len` bytes from the next unclaimed one; `None` when the
    /// range is finished.
    pub fn claim(&self, len: u64) -> Option<(u64, u64)> {
        let mut state = self.state.lock();
        if state.next > state.end || len == 0 {
            return None;
        }
        let take = len.min(state.end + 1 - state.next);
        let offset = state.next;
        state.next += take;
        Some((offset, take))
    }

    /// Give back a claim whose write failed.
    pub fn unclaim(&self, offset: u64) {
        let mut state = self.state.lock();
        state.next = state.next.min(offset);
    }

    fn remaining(&self) -> u64 {
        let state = self.state.lock();
        (state.end + 1).saturating_sub(state.next)
    }
}

/// The ranges of one file being fetched right now, so an idle connection
/// can take half of the largest instead of waiting for it to finish.
#[derive(Debug)]
pub struct InFlight {
    min_split: u64,
    cursors: Mutex<Vec<Arc<RangeCursor>>>,
}

impl InFlight {
    /// Ranges split no smaller than `min_split` bytes.
    pub fn new(min_split: u64) -> Self {
        Self { min_split: min_split.max(1), cursors: Mutex::new(Vec::new()) }
    }

    /// Track `cursor` until [`Self::finish`].
    pub fn start(&self, cursor: &Arc<RangeCursor>) {
        self.cursors.lock().push(Arc::clone(cursor));
    }

    /// Stop tracking `cursor`.
    pub fn finish(&self, cursor: &Arc<RangeCursor>) {
        self.cursors.lock().retain(|c| !Arc::ptr_eq(c, cursor));
    }

    /// Whether some range has enough left to split.
    pub fn splittable(&self) -> bool {
        self.cursors.lock().iter().any(|c| c.remaining() >= 2 * self.min_split)
    }

    /// Take the upper half of the range with the most bytes left, as a new
    /// tracked range.  `None` when nothing has enough left to split.
    pub fn split(&self) -> Option<Arc<RangeCursor>> {
        let mut cursors = self.cursors.lock();
        let largest = cursors.iter().max_by_key(|c| c.remaining())?;
        let taken = {
            let mut state = largest.state.lock();
            let remaining = (state.end + 1).saturating_sub(state.next);
            if remaining < 2 * self.min_split {
                return None;
            }
            let mid = state.next + remaining / 2;
            let taken = RangeCursor::new(mid, state.end);
            state.end = mid - 1;
            taken
        };
        cursors.push(Arc::clone(&taken));
        Some(taken)
    }
}

/// How long the tuner watches one connection count before deciding.
pub const PROBE_WINDOW: Duration = Duration::from_secs(10);
/// Connections added or removed per step.
pub const TUNE_STEP: usize = 2;
/// Growth the last step must have bought to keep growing.
pub const MIN_GAIN: f64 = 0.10;
/// Below this many bytes per second per connection the link is split so
/// thin that more connections only lengthen the tail.
pub const CROWDED_RATE: f64 = 512.0 * 1024.0;
/// Clean windows after a back-off before the tuner probes upward again: a
/// blip early in a long download shouldn't cap it for the rest.
pub const REGROW_AFTER: u32 = 3;

/// What the tuner decided after a probe window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuneStep {
    pub from: usize,
    pub to: usize,
    /// Total bytes per second over the window.
    pub rate: f64,
}

/// Moves a download's connection count toward what its link can use: grow
/// while each step still raises total throughput, hold once it stops, shrink
/// when the per-connection rate shows the link is crowded, and back off when
/// the server pushes back (probing upward again once it has been quiet for
/// [`REGROW_AFTER`] windows).
#[derive(Debug)]
pub struct ConnectionTuner {
    window_started: Duration,
    received_at_start: u64,
    /// Total rate of the previous window, to judge the last step by.
    last_rate: Option<f64>,
    growing: bool,
    throttled: bool,
    errors: u32,
    /// Clean windows since the last back-off, while waiting to regrow.
    since_backoff: Option<u32>,
}

impl ConnectionTuner {
    /// A tuner whose first window starts at `now` with `received` bytes in.
    pub fn new(now: Duration, received: u64) -> Self {
        Self { window_started: now, received_at_start: received, last_rate: None, growing: true, throttled: false, errors: 0, since_backoff: None }
    }

    /// The server answered 429 or 503.
    pub fn note_throttled(&mut self) {
        self.throttled = true;
    }

    /// A connection failed (not counting deliberate low-speed drops).
    pub fn note_error(&mut self) {
        self.errors += 1;
    }

    /// Decide after a full [`PROBE_WINDOW`]: `Some` when the limit should
    /// change from `limit`, `None` to keep it (or while the window runs).
    pub fn step(&mut self, now: Duration, received: u64, limit: usize) -> Option<TuneStep> {
        let elapsed = now.saturating_sub(self.window_started);
        if elapsed < PROBE_WINDOW {
            return None;
        }
        let rate = received.saturating_sub(self.received_at_start) as f64 / elapsed.as_secs_f64();
        let pushed_back = self.throttled || self.errors as usize >= (limit / 2).max(2);
        self.window_started = now;
        self.received_at_start = received;
        self.throttled = false;
        self.errors = 0;

        let fewer = limit.saturating_sub(TUNE_STEP).max(MIN_CONNECTIONS).min(limit);
        if pushed_back {
            self.since_backoff = Some(0);
        } else if let Some(clean) = self.since_backoff.as_mut() {
            *clean += 1;
            if *clean >= REGROW_AFTER {
                self.since_backoff = None;
                self.growing = true;
                self.last_rate = None;
            }
        }
        let to = if pushed_back || rate / (limit as f64) < CROWDED_RATE {
            self.growing = false;
            fewer
        } else if self.growing && self.last_rate.is_none_or(|last| rate >= last * (1.0 + MIN_GAIN)) {
            (limit + TUNE_STEP).min(MAX_CONNECTIONS)
        } else {
            self.growing = false;
            limit
        };
        self.last_rate = Some(rate);
        (to != limit).then_some(TuneStep { from: limit, to, rate })
    }
}

/// A download's shared pool with the tuner that sizes it.
pub struct Tuning {
    pub permits: Arc<ConnectionPermits>,
    pub stats: Arc<TransferStats>,
    tuner: Mutex<ConnectionTuner>,
}

impl Tuning {
    /// Tune `permits` from `stats`, starting now.
    pub fn new(permits: Arc<ConnectionPermits>, stats: Arc<TransferStats>) -> Arc<Self> {
        let tuner = ConnectionTuner::new(stats.now(), stats.received());
        Arc::new(Self { permits, stats, tuner: Mutex::new(tuner) })
    }

    /// Step the tuner if its window is up and nobody else is; returns the
    /// change made, if any.
    pub fn tick(&self) -> Option<TuneStep> {
        let mut tuner = self.tuner.try_lock()?;
        let step = tuner.step(self.stats.now(), self.stats.received(), self.permits.limit())?;
        self.permits.set_limit(step.to);
        Some(step)
    }

    /// The server answered 429 or 503.
    pub fn note_throttled(&self) {
        self.tuner.lock().note_throttled();
    }

    /// A connection failed.
    pub fn note_error(&self) {
        self.tuner.lock().note_error();
    }
}
