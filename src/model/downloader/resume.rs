//! What a chunked download has already written, so it can pick up where it
//! stopped after a failure or a restart.

use std::collections::BTreeMap;

/// A set of byte offsets as merged half-open intervals `[start, end)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntervalSet {
    /// start -> end; never overlapping or touching.
    spans: BTreeMap<u64, u64>,
}

impl IntervalSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// A set holding each `[start, end)` in `spans`.
    pub fn from_spans(spans: impl IntoIterator<Item = (u64, u64)>) -> Self {
        let mut set = Self::new();
        for (start, end) in spans {
            set.insert(start, end);
        }
        set
    }

    /// Add `[start, end)`, merging with whatever it overlaps or touches.
    pub fn insert(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        let (mut start, mut end) = (start, end);
        // A span starting at or before `start` that reaches it.
        if let Some((&s, &e)) = self.spans.range(..=start).next_back() {
            if e >= start {
                start = s;
                end = end.max(e);
            }
        }
        // Every span starting inside the new one is swallowed.
        let swallowed: Vec<u64> = self.spans.range(start..=end).map(|(&s, _)| s).collect();
        for s in swallowed {
            if let Some(e) = self.spans.remove(&s) {
                end = end.max(e);
            }
        }
        self.spans.insert(start, end);
    }

    /// Total bytes covered.
    pub fn len_bytes(&self) -> u64 {
        self.spans.iter().map(|(s, e)| e - s).sum()
    }

    /// True when nothing is covered.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// True when `[0, total)` is covered entirely.
    pub fn covers(&self, total: u64) -> bool {
        total == 0 || self.spans.get(&0).is_some_and(|&e| e >= total)
    }

    /// The gaps in `[0, total)`, as half-open intervals in order.
    pub fn missing(&self, total: u64) -> Vec<(u64, u64)> {
        let mut gaps = Vec::new();
        let mut cursor = 0u64;
        for (&s, &e) in &self.spans {
            if s >= total {
                break;
            }
            if s > cursor {
                gaps.push((cursor, s));
            }
            cursor = cursor.max(e);
        }
        if cursor < total {
            gaps.push((cursor, total));
        }
        gaps
    }

    /// The covered spans, in order.
    pub fn spans(&self) -> Vec<(u64, u64)> {
        self.spans.iter().map(|(&s, &e)| (s, e)).collect()
    }
}
