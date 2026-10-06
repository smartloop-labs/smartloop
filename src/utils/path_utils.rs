//! Port of `smartloop/utils/path_utils.py`.
//!
//! Path utilities for cross-platform (WSL / Windows / Linux) compatibility.

/// Port of `ntpath.splitroot` on a path whose `/` separators were already
/// normalised to `\` (as `PureWindowsPath` does): returns `(drive, root, rest)`.
fn nt_splitroot(p: &[char]) -> (String, String, String) {
    const SEP: char = '\\';
    let s = |a: usize, b: usize| -> String { p[a.min(p.len())..b.min(p.len())].iter().collect() };
    let find = |start: usize| -> Option<usize> { (start..p.len()).find(|&i| p[i] == SEP) };
    let len = p.len();

    if p.first() == Some(&SEP) {
        if p.get(1) == Some(&SEP) {
            // UNC drives, e.g. \\server\share or \\?\UNC\server\share
            // Device drives, e.g. \\.\device or \\?\device
            let unc_prefix = "\\\\?\\UNC\\";
            let start = if s(0, 8).to_uppercase() == unc_prefix { 8 } else { 2 };
            let Some(index) = find(start) else {
                return (s(0, len), String::new(), String::new());
            };
            let Some(index2) = find(index + 1) else {
                return (s(0, len), String::new(), String::new());
            };
            (s(0, index2), s(index2, index2 + 1), s(index2 + 1, len))
        } else {
            // Relative path with root, e.g. \Windows
            (String::new(), s(0, 1), s(1, len))
        }
    } else if p.get(1) == Some(&':') {
        if p.get(2) == Some(&SEP) {
            // Absolute drive-letter path, e.g. X:\Windows
            (s(0, 2), s(2, 3), s(3, len))
        } else {
            // Relative path with drive, e.g. X:Windows
            (s(0, 2), String::new(), s(2, len))
        }
    } else {
        (String::new(), String::new(), s(0, len))
    }
}

/// Convert a Windows-style path to a WSL path when running on Linux.
///
/// On native Windows the path is returned unchanged.  On WSL/Linux a
/// client-supplied path like `C:\Users\foo\bar` is translated to
/// `/mnt/c/Users/foo/bar` so that it resolves correctly on the host.
pub fn normalize_path(path: &str) -> String {
    if cfg!(windows) {
        return path.to_string();
    }
    let normalized: Vec<char> = path.replace('/', "\\").chars().collect();
    let (drive, _root, rest) = nt_splitroot(&normalized);
    if drive.is_empty() {
        return path.to_string();
    }
    let drive_letter: String = drive.chars().next().map(|c| c.to_lowercase().collect()).unwrap_or_default();
    // PureWindowsPath.parts[1:] — the components after the drive/root anchor,
    // with empty and "." components dropped.
    let mut out = format!("/mnt/{drive_letter}");
    for part in rest.split('\\').filter(|x| !x.is_empty() && *x != ".") {
        out.push('/');
        out.push_str(part);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(windows))]
    #[test]
    fn translates_drive_paths() {
        assert_eq!(normalize_path(r"C:\Users\foo\bar"), "/mnt/c/Users/foo/bar");
        assert_eq!(normalize_path("D:/data/x.txt"), "/mnt/d/data/x.txt");
        assert_eq!(normalize_path("C:"), "/mnt/c");
        assert_eq!(normalize_path(r"C:\"), "/mnt/c");
        assert_eq!(normalize_path(r"C:foo\.\bar"), "/mnt/c/foo/bar");
        assert_eq!(normalize_path("/home/user/file"), "/home/user/file");
        assert_eq!(normalize_path("relative/path"), "relative/path");
    }
}
