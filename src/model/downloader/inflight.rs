//! Port of `smartloop/model/downloader/inflight.py`.
//!
//! Which model directories a download is currently writing to.
//!
//! On disk a live download and one that died look identical: a directory
//! holding a `.part` and no finished GGUF.  The storage sweep cannot tell them
//! apart, so it read a transfer in progress as abandoned and deleted it out
//! from under the downloader.  This module carries the one fact the filesystem
//! does not: someone is still writing.
//!
//! Process-local and advisory, deliberately.  No lock is held across a
//! transfer, so consulting the registry costs microseconds and a cleanup never
//! blocks behind a multi-GB download.

use std::collections::HashMap;
use std::path::Path;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

/// Canonical directory path -> how many transfers are writing to it.  A count
/// rather than a set: the same model can be fetched twice concurrently and the
/// first to finish must not unmark the directory while the second still writes.
static ACTIVE: Lazy<Mutex<HashMap<String, usize>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Canonical form of `path`, as both sides of the registry must agree.
///
/// Port of `os.path.normcase(os.path.realpath(path))`: resolves symlinks (on
/// macOS a temp directory under `/var` resolves to `/private/var`) even for
/// paths that do not exist yet.
fn key(path: &Path) -> String {
    let _ = path;
    todo!()
}

/// RAII marker returned by [`downloading`]; unmarks the directory on drop
/// (the `finally` of the Python context manager).
#[derive(Debug)]
pub struct DownloadingGuard {
    key: String,
}

impl Drop for DownloadingGuard {
    fn drop(&mut self) {
        let _ = &self.key;
        // Body filled in during the implementation phase.
    }
}

/// Mark `local_dir` as being written to for as long as the guard lives.
pub fn downloading(local_dir: impl AsRef<Path>) -> DownloadingGuard {
    let _ = (key(local_dir.as_ref()), &*ACTIVE);
    todo!()
}

/// True while a transfer is writing at, below, or above `path`.
///
/// Containment in both directions, because the caller is asking "is it safe to
/// delete this" and both answers are no.
pub fn is_downloading(path: impl AsRef<Path>) -> bool {
    let _ = path;
    todo!()
}

/// Snapshot of the registry (`_active` in Python), for tests.
pub fn active_snapshot() -> HashMap<String, usize> {
    todo!()
}

/// Drop every mark (`_active.clear()` in the Python tests).  Tests only.
pub fn reset_for_tests() {
    todo!()
}
