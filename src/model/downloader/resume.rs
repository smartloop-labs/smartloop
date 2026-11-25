//! What a chunked download has already written, so it can pick up where it
//! stopped after a failure or a restart.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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

/// Format version of the sidecar; a file with any other version is ignored.
pub const RESUME_VERSION: u32 = 1;

/// The sidecar recording a chunked download's progress, next to its `.part`:
/// the object it belongs to and the byte spans already durable on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResumeState {
    pub version: u32,
    pub size: u64,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    /// Half-open `[start, end)` spans.
    pub done: Vec<[u64; 2]>,
}

/// `<part>.state`.
pub fn state_path(part: &Path) -> PathBuf {
    let mut name = part.as_os_str().to_owned();
    name.push(".state");
    PathBuf::from(name)
}

/// Whether an ETag is a weak validator (`W/"…"`), which `If-Range` rejects.
pub fn is_weak_etag(etag: &str) -> bool {
    etag.trim_start().starts_with("W/")
}

impl ResumeState {
    /// A state for an object with no bytes written yet.
    pub fn new(size: u64, etag: Option<String>, last_modified: Option<String>) -> Self {
        Self { version: RESUME_VERSION, size, etag, last_modified, done: Vec::new() }
    }

    /// Whether this object can be resumed at all: it needs a validator to
    /// tell a later run that the remote bytes are still the same.
    pub fn resumable(&self) -> bool {
        self.size > 0 && (self.etag.as_deref().is_some_and(|e| !is_weak_etag(e)) || self.last_modified.is_some())
    }

    /// Whether this state describes the object a probe just saw.  The ETag
    /// decides when both sides have one; Last-Modified only stands in when
    /// neither does.
    pub fn same_object(&self, size: u64, etag: Option<&str>, last_modified: Option<&str>) -> bool {
        if self.size != size || !self.resumable() {
            return false;
        }
        match (self.etag.as_deref(), etag) {
            (Some(ours), Some(theirs)) => ours == theirs && !is_weak_etag(ours),
            (None, None) => self.last_modified.is_some() && self.last_modified.as_deref() == last_modified,
            _ => false,
        }
    }

    /// The spans already written.
    pub fn done_set(&self) -> IntervalSet {
        IntervalSet::from_spans(self.done.iter().map(|[s, e]| (*s, *e)))
    }

    /// Replace the written spans.
    pub fn set_done(&mut self, done: &IntervalSet) {
        self.done = done.spans().into_iter().map(|(s, e)| [s, e]).collect();
    }

    /// The sidecar for `part`, or `None` when it is missing, unreadable or
    /// from another format version.
    pub fn load(part: &Path) -> Option<Self> {
        let bytes = std::fs::read(state_path(part)).ok()?;
        let state: Self = serde_json::from_slice(&bytes).ok()?;
        (state.version == RESUME_VERSION).then_some(state)
    }

    /// Write the sidecar for `part` atomically (temp file, fsync, rename).
    pub fn save(&self, part: &Path) -> std::io::Result<()> {
        let path = state_path(part);
        let mut tmp_name = path.as_os_str().to_owned();
        tmp_name.push(".tmp");
        let tmp = PathBuf::from(tmp_name);
        {
            let mut file = std::fs::File::create(&tmp)?;
            file.write_all(&serde_json::to_vec(self).map_err(std::io::Error::other)?)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp, &path)
    }

    /// Remove the sidecar for `part`, if there is one.
    pub fn remove(part: &Path) -> std::io::Result<()> {
        match std::fs::remove_file(state_path(part)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }
}
