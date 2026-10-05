//! Process-level environment fixes applied once at startup.
//!
//! Replaces the PyInstaller runtime hooks the Python build shipped
//! (`hooks/rthook_utf8.py`, `hooks/rthook_vulkan_icd.py`).  Those ran before
//! any application import; the Rust equivalent is [`apply_runtime_env`], which
//! `main()` must call first thing — before the tokio runtime or any other
//! thread is started, because it may mutate the process environment.

use std::path::{Path, PathBuf};

/// File name of the MoltenVK ICD manifest a packaged macOS build ships when it
/// is built with the Vulkan backend (rewritten so `library_path` points at the
/// bundled `libMoltenVK.dylib`).
pub const MOLTENVK_ICD_FILE: &str = "MoltenVK_icd.json";

/// Directory next to the `slp` executable that holds bundled shared libraries
/// (Vulkan loader, MoltenVK) in a packaged build.
pub const BUNDLED_LIB_SUBDIR: &str = "lib";

/// Apply every startup environment fix for the current platform.
///
/// * Windows: switch the attached console to UTF-8 (code page 65001).  Rust
///   itself reads files as bytes and writes to a console via the wide-char
///   API, so the `Path.read_text()` half of the old hook has no equivalent
///   failure mode here; but text that native code (llama.cpp/ggml logging)
///   writes straight to the console would otherwise be rendered in the OEM
///   code page and mangle characters such as `→`.  Failures are ignored —
///   there may be no console at all (service, redirected output).
/// * macOS: point the Vulkan loader at the bundled MoltenVK ICD, unless the
///   user already configured a driver via `VK_DRIVER_FILES` /
///   `VK_ICD_FILENAMES`.  The loader otherwise only finds drivers in system
///   paths (`/usr/local/share/vulkan/icd.d`, ...) that exist only if the user
///   installed MoltenVK themselves.  A Metal-only build ships no manifest, in
///   which case this is a no-op.
/// * Linux: nothing to do — the GPU driver registers its own ICD.
///
/// Must be called before any other thread is spawned.
pub fn apply_runtime_env() {
    #[cfg(windows)]
    enable_utf8_console();

    #[cfg(target_os = "macos")]
    if let Some(exe_dir) = executable_dir() {
        let configured = |name: &str| std::env::var_os(name).is_some_and(|v| !v.is_empty());
        if let Some(icd) = vulkan_icd_override(&exe_dir, configured) {
            // SAFETY: documented contract — called at the very start of
            // `main()`, before any other thread exists.
            unsafe {
                std::env::set_var("VK_DRIVER_FILES", &icd);
                // Older loaders only honor the legacy name.
                std::env::set_var("VK_ICD_FILENAMES", &icd);
            }
        }
    }
}

/// Directory containing the running executable, with symlinks resolved.
///
/// The installers expose `slp` through symlinks (`~/.smartloop/slp`,
/// `~/.local/bin/slp`) into the versioned install dir, and bundled files live
/// next to the real binary, so the path is canonicalized.
pub fn executable_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
    exe.parent().map(Path::to_path_buf)
}

/// The bundled MoltenVK ICD manifest under `exe_dir`, if one exists.
///
/// Looks in `<exe_dir>/lib/` (where the Makefile stages it) and then directly
/// in `<exe_dir>`.
pub fn bundled_moltenvk_icd(exe_dir: &Path) -> Option<PathBuf> {
    [
        exe_dir.join(BUNDLED_LIB_SUBDIR).join(MOLTENVK_ICD_FILE),
        exe_dir.join(MOLTENVK_ICD_FILE),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

/// Decide whether the Vulkan loader should be pointed at a bundled ICD.
///
/// `is_configured(name)` reports whether env var `name` is set to a non-empty
/// value.  Returns the manifest path to export as `VK_DRIVER_FILES` /
/// `VK_ICD_FILENAMES`, or `None` when the user already configured a driver or
/// no manifest is bundled.
pub fn vulkan_icd_override(
    exe_dir: &Path,
    is_configured: impl Fn(&str) -> bool,
) -> Option<PathBuf> {
    if is_configured("VK_DRIVER_FILES") || is_configured("VK_ICD_FILENAMES") {
        return None;
    }
    bundled_moltenvk_icd(exe_dir)
}

#[cfg(windows)]
fn enable_utf8_console() {
    const CP_UTF8: u32 = 65001;
    unsafe extern "system" {
        fn SetConsoleOutputCP(code_page: u32) -> i32;
        fn SetConsoleCP(code_page: u32) -> i32;
    }
    // SAFETY: plain Win32 calls with a constant argument; a zero return (no
    // console attached) is deliberately ignored.
    unsafe {
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_manifest_means_no_override() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(vulkan_icd_override(dir.path(), |_| false), None);
    }

    #[test]
    fn bundled_manifest_in_lib_dir_is_used() {
        let dir = tempfile::tempdir().unwrap();
        let lib = dir.path().join(BUNDLED_LIB_SUBDIR);
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join(MOLTENVK_ICD_FILE), "{}").unwrap();
        assert_eq!(
            vulkan_icd_override(dir.path(), |_| false),
            Some(lib.join(MOLTENVK_ICD_FILE))
        );
    }

    #[test]
    fn manifest_next_to_binary_is_used() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(MOLTENVK_ICD_FILE), "{}").unwrap();
        assert_eq!(
            vulkan_icd_override(dir.path(), |_| false),
            Some(dir.path().join(MOLTENVK_ICD_FILE))
        );
    }

    #[test]
    fn user_configured_driver_wins() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(MOLTENVK_ICD_FILE), "{}").unwrap();
        assert_eq!(
            vulkan_icd_override(dir.path(), |n| n == "VK_DRIVER_FILES"),
            None
        );
        assert_eq!(
            vulkan_icd_override(dir.path(), |n| n == "VK_ICD_FILENAMES"),
            None
        );
    }
}
