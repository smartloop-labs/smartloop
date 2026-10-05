//! Port of `smartloop/utils/path_utils.py`.
//!
//! Path utilities for cross-platform (WSL / Windows / Linux) compatibility.

/// Convert a Windows-style path to a WSL path when running on Linux.
///
/// On native Windows the path is returned unchanged.  On WSL/Linux a
/// client-supplied path like `C:\Users\foo\bar` is translated to
/// `/mnt/c/Users/foo/bar` so that it resolves correctly on the host.
pub fn normalize_path(path: &str) -> String {
    let _ = path;
    todo!()
}
