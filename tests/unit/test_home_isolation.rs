//! Guards the process-wide test isolation in tests/common/mod.rs: no test may
//! resolve the developer's real `~/.smartloop`.

use std::ffi::CStr;

use smartloop::config::AppSettings;
use smartloop::workspace::workspace::Workspace;

/// The home directory from the passwd database, which no env var can move.
fn passwd_home() -> String {
    // SAFETY: getpwuid returns a pointer into static storage, read at once.
    unsafe {
        let pw = libc::getpwuid(libc::getuid());
        assert!(!pw.is_null(), "no passwd entry for the current user");
        CStr::from_ptr((*pw).pw_dir).to_string_lossy().into_owned()
    }
}

#[test]
fn test_slp_home_is_not_the_real_home() {
    let real = format!("{}/.smartloop", passwd_home());
    let home = AppSettings::default().home_dir();
    assert_ne!(home, real, "tests must never resolve the real ~/.smartloop");
    assert!(home.contains("slp-test-home-"), "unexpected test home: {home}");
}

#[test]
fn test_workspace_config_file_is_isolated() {
    let real = passwd_home();
    let config = Workspace::config_file();
    assert!(!config.starts_with(&real), "config.json resolved under the real home: {config}");
}
