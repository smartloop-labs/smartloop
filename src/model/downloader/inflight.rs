//! Port of `smartloop/model/downloader/inflight.py`.
//!
//! Which model directories a download is currently writing to.
//!
//! On disk a live download and one that died look identical: a directory
//! holding a `.part` and no finished GGUF.  `storage_service::weights_state`
//! cannot tell them apart, so the disk sweep read a transfer in progress as
//! abandoned and deleted it out from under the downloader.  This module
//! carries the one fact the filesystem does not: someone is still writing.
//!
//! Process-local and advisory, deliberately.  No lock is held across a
//! transfer, so consulting the registry costs microseconds and a cleanup never
//! blocks behind a multi-GB download.  It is also why the state lives here
//! rather than in a database: "a transfer is running" is true exactly as long
//! as a thread is, and a row that outlived the process would claim a dead
//! download was live forever — wedging cleanup on a full disk, which is when
//! it matters most.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf, MAIN_SEPARATOR};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

/// Canonical directory path -> how many transfers are writing to it.  A count
/// rather than a set: nothing serializes downloads on model_id, so the same
/// model can be fetched twice concurrently and the first to finish must not
/// unmark the directory while the second is still writing.
static ACTIVE: Lazy<Mutex<HashMap<String, usize>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Maximum symlink hops followed while resolving one path (guards loops).
const MAX_SYMLINK_HOPS: usize = 40;

/// Port of `os.path.realpath`: make `path` absolute, then walk it component by
/// component, following every symlink that exists and resolving `..`
/// lexically against what has been resolved so far.  Components that do not
/// exist are appended as-is, so a directory that is about to be created
/// canonicalises the same way it will once it exists.
fn realpath(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().map(|cwd| cwd.join(path)).unwrap_or_else(|_| path.to_path_buf())
    };
    let mut hops = 0usize;
    resolve_into(PathBuf::new(), &absolute, &mut hops)
}

fn resolve_into(mut resolved: PathBuf, path: &Path, hops: &mut usize) -> PathBuf {
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => {
                resolved.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                let candidate = resolved.join(name);
                match std::fs::symlink_metadata(&candidate) {
                    Ok(meta) if meta.file_type().is_symlink() && *hops < MAX_SYMLINK_HOPS => {
                        *hops += 1;
                        match std::fs::read_link(&candidate) {
                            Ok(target) => {
                                resolved = if target.is_absolute() {
                                    resolve_into(PathBuf::new(), &target, hops)
                                } else {
                                    resolve_into(resolved, &target, hops)
                                };
                            }
                            Err(_) => resolved = candidate,
                        }
                    }
                    _ => resolved = candidate,
                }
            }
        }
    }
    resolved
}

/// Port of `os.path.normcase`: identity on POSIX (macOS included), lower-case
/// with forward slashes turned into backslashes on Windows.
fn normcase(path: String) -> String {
    if cfg!(windows) { path.replace('/', "\\").to_lowercase() } else { path }
}

/// Canonical form of `path`, as both sides of the registry must agree.
///
/// `realpath` is not cosmetic here: the marker is written with the
/// downloader's `local_dir` and read with the sweep's scan path, and on macOS
/// a temp directory under `/var` resolves to `/private/var`.  `normcase`
/// covers callers on a case-insensitive filesystem.
fn key(path: &Path) -> String {
    normcase(realpath(path).to_string_lossy().into_owned())
}

/// RAII marker returned by [`downloading`]; unmarks the directory on drop
/// (the `finally` of the Python context manager).
#[derive(Debug)]
pub struct DownloadingGuard {
    key: String,
}

impl Drop for DownloadingGuard {
    fn drop(&mut self) {
        let mut active = ACTIVE.lock();
        let remaining = active.get(&self.key).copied().unwrap_or(0).saturating_sub(1);
        if remaining > 0 {
            active.insert(self.key.clone(), remaining);
        } else {
            active.remove(&self.key);
        }
    }
}

/// Mark `local_dir` as being written to for as long as the guard lives.
pub fn downloading(local_dir: impl AsRef<Path>) -> DownloadingGuard {
    let key = key(local_dir.as_ref());
    *ACTIVE.lock().entry(key.clone()).or_insert(0) += 1;
    DownloadingGuard { key }
}

/// True while a transfer is writing at, below, or above `path`.
///
/// Containment in both directions, because the caller is asking "is it safe to
/// delete this" and both answers are no.  Below: removing `models/<org>`
/// destroys a download into `models/<org>/<repo>`.  Above: removing
/// `<model_dir>/bin` takes files out of a live transfer — weights can land
/// there, as `weights_state` and the init router's `gguf_exists` both know.
/// The mapping holds one entry per running transfer, so the scan is free
/// either way.
pub fn is_downloading(path: impl AsRef<Path>) -> bool {
    let key = key(path.as_ref());
    let sep = MAIN_SEPARATOR.to_string();
    let key_prefix = format!("{key}{sep}");
    let active = ACTIVE.lock();
    active.keys().any(|marked| {
        *marked == key
            || marked.starts_with(&key_prefix) // a transfer below *path*
            || key.starts_with(&format!("{marked}{sep}")) // *path* inside a transfer
    })
}

/// Snapshot of the registry (`_active` in Python), for tests.
pub fn active_snapshot() -> HashMap<String, usize> {
    ACTIVE.lock().clone()
}

/// Drop every mark (`_active.clear()` in the Python tests).  Tests only.
pub fn reset_for_tests() {
    ACTIVE.lock().clear();
}
