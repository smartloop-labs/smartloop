//! Port of `tests/unit/test_adopt_confined_home.py`.
//!
//! An install that kept its state in a store sandbox, brought forward.
//!
//! The framework home is `~/.smartloop` in every build, so the app and the CLI
//! share one workspace.  Builds that shipped before the home was resolved past
//! the sandbox put it under the rewritten `$HOME` instead, and those installs
//! still have their workspace, projects and model weights in there: the move
//! has to happen on their behalf, before anything reads the home, or an
//! existing user starts the new build to an empty workspace.

use std::fs;
use std::path::{Path, PathBuf};

use serial_test::serial;

use smartloop::workspace::migrations::{ADOPTED_MARKER, adopt_confined_home};

use crate::shared::workspace_support::{EnvGuard, ProcessScopeReset};

const CONTAINER: &str = "Library/Containers/ai.smartloop.studio/Data";

fn s(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// The `homes` fixture: a container-confined `$HOME` and the real home beside
/// it.
struct Homes {
    old: PathBuf,
    dest: PathBuf,
    _tmp: tempfile::TempDir,
    _env: Vec<EnvGuard>,
    _reset: ProcessScopeReset,
}

fn homes() -> Homes {
    let reset = ProcessScopeReset::new();
    let tmp = tempfile::tempdir().unwrap();
    let mut env = vec![
        EnvGuard::remove("SLP_HOME"),
        EnvGuard::remove("SNAP_NAME"),
        EnvGuard::remove("SNAP_REAL_HOME"),
    ];

    let real = tmp.path().join("Users").join("tester");
    let container = real.join(CONTAINER);
    let old = container.join(".smartloop");
    fs::create_dir_all(old.join("workspace-1").join("models")).unwrap();
    fs::write(old.join("config.json"), r#"{"id": "workspace-1"}"#).unwrap();
    fs::write(old.join(".credentials"), "secret").unwrap();
    fs::write(old.join("workspace-1").join("models").join("weights.gguf"), "x".repeat(32)).unwrap();

    env.push(EnvGuard::set("HOME", &container));
    let dest = real.join(".smartloop");
    Homes { old, dest, _tmp: tmp, _env: env, _reset: reset }
}

mod bringing_the_state_forward {
    use super::*;

    #[test]
    #[serial]
    fn test_the_whole_home_moves_to_the_real_one() {
        let homes = homes();
        assert_eq!(adopt_confined_home(&s(&homes.dest)), Some(s(&homes.old)));

        let dest = &homes.dest;
        assert_eq!(fs::read_to_string(dest.join("config.json")).unwrap(), r#"{"id": "workspace-1"}"#);
        assert_eq!(fs::read_to_string(dest.join(".credentials")).unwrap(), "secret");
        // Model weights run to gigabytes; they travel with everything else.
        assert!(dest.join("workspace-1").join("models").join("weights.gguf").exists());
    }

    #[test]
    #[serial]
    fn test_the_old_home_is_emptied_and_marked() {
        let homes = homes();
        adopt_confined_home(&s(&homes.dest));
        let old = &homes.old;
        assert!(!old.join("config.json").exists());
        assert_eq!(fs::read_to_string(old.join(ADOPTED_MARKER)).unwrap().trim(), s(&homes.dest));
    }

    #[test]
    #[serial]
    fn test_a_second_start_does_nothing() {
        let homes = homes();
        adopt_confined_home(&s(&homes.dest));
        // Whatever the running install has written since must not be walked
        // over by a migration that already happened.
        fs::write(homes.dest.join("config.json"), r#"{"id": "moved-on"}"#).unwrap();
        assert_eq!(adopt_confined_home(&s(&homes.dest)), None);
        assert_eq!(fs::read_to_string(homes.dest.join("config.json")).unwrap(), r#"{"id": "moved-on"}"#);
    }
}

mod when_it_declines {
    use super::*;

    #[test]
    #[serial]
    fn test_an_unconfined_home_is_not_a_migration() {
        let _reset = ProcessScopeReset::new();
        let tmp = tempfile::tempdir().unwrap();
        let _slp_home = EnvGuard::remove("SLP_HOME");
        let _home = EnvGuard::set("HOME", tmp.path());
        fs::create_dir(tmp.path().join(".smartloop")).unwrap();
        assert_eq!(adopt_confined_home(&s(&tmp.path().join(".smartloop"))), None);
    }

    #[test]
    #[serial]
    fn test_an_explicit_slp_home_outranks_it() {
        let homes = homes();
        // A service unit, a test, or a packaged build that means to stay in
        // its container said where the home is. That instruction wins.
        let _slp_home = EnvGuard::set("SLP_HOME", &homes.old);
        assert_eq!(adopt_confined_home(&s(&homes.dest)), None);
        assert!(homes.old.join("config.json").exists());
    }

    #[test]
    #[serial]
    fn test_a_workspace_already_at_the_real_home_is_left_alone() {
        let homes = homes();
        // Two live workspaces: merging them would be a guess about whose
        // project list is the real one, so the user keeps both.
        fs::create_dir_all(&homes.dest).unwrap();
        fs::write(homes.dest.join("config.json"), r#"{"id": "the-cli-one"}"#).unwrap();

        assert_eq!(adopt_confined_home(&s(&homes.dest)), None);
        assert_eq!(fs::read_to_string(homes.dest.join("config.json")).unwrap(), r#"{"id": "the-cli-one"}"#);
        assert!(homes.old.join("config.json").exists());
    }

    #[test]
    #[serial]
    fn test_nothing_to_bring_forward() {
        let _reset = ProcessScopeReset::new();
        let tmp = tempfile::tempdir().unwrap();
        let _slp_home = EnvGuard::remove("SLP_HOME");
        let container = tmp.path().join("Users").join("tester").join(CONTAINER);
        fs::create_dir_all(&container).unwrap();
        let _home = EnvGuard::set("HOME", &container);
        assert_eq!(
            adopt_confined_home(&s(&tmp.path().join("Users").join("tester").join(".smartloop"))),
            None
        );
    }
}

mod a_snap_install {
    use super::*;

    #[test]
    #[serial]
    fn test_the_revision_directory_is_brought_forward_too() {
        let _reset = ProcessScopeReset::new();
        let tmp = tempfile::tempdir().unwrap();
        let _slp_home = EnvGuard::remove("SLP_HOME");
        let _snap_real_home = EnvGuard::remove("SNAP_REAL_HOME");
        let real = tmp.path().join("home").join("tester");
        let confined = real.join("snap").join("smartloop").join("42");
        let old = confined.join(".smartloop");
        fs::create_dir_all(&old).unwrap();
        fs::write(old.join("config.json"), r#"{"id": "workspace-1"}"#).unwrap();

        let _home = EnvGuard::set("HOME", &confined);
        let _snap_name = EnvGuard::set("SNAP_NAME", "smartloop");

        let dest = real.join(".smartloop");
        assert_eq!(adopt_confined_home(&s(&dest)), Some(s(&old)));
        assert_eq!(fs::read_to_string(dest.join("config.json")).unwrap(), r#"{"id": "workspace-1"}"#);
    }
}
