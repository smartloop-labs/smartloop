//! Port of `tests/api/test_init_needs_whole_weights.py`.
//!
//! `/v1/init` must not call a half-downloaded model ready.
//!
//! `gguf_exists` decides whether the route downloads at all, so whatever it
//! accepts is reported as available and never revisited.  It used to accept
//! any `.gguf`, which an interrupted multi-file download satisfies with the
//! wrong file: the companion that happened to finish first.
//!
//! Observed live — a restart during the final verify left `mmproj` renamed and
//! the 2 GB weights as `.part`.  Every later `/v1/init` answered "already
//! available", so the weights could never arrive; the workspace recorded the
//! model as downloaded and the client showed it ready.

use std::path::Path;

use smartloop::api::routers::init::gguf_exists;
use tempfile::TempDir;

const WEIGHTS: &str = "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf";
const MMPROJ: &str = "mmproj-Ministral-3-3B-Instruct-2512.gguf";

/// Somewhere to plant fake weights (the `weights_root` fixture): a temporary
/// directory of its own, so the files planted here are never mistaken for a
/// download that escaped the stub.
fn weights_root() -> TempDir {
    tempfile::Builder::new().prefix("weights").tempdir().unwrap()
}

/// Create `<root>/google/some-model` holding empty files at each of
/// `relative_names`, and return its path.
fn model_dir(weights_root: &Path, relative_names: &[&str]) -> String {
    let d = weights_root.join("google").join("some-model");
    for rel in relative_names {
        let p = d.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"").unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().into_owned()
}

/// `TestAnUnfinishedTransfer` — a `.part` is the downloader saying a file is
/// not finished.
mod test_an_unfinished_transfer {
    use super::*;

    /// mmproj renamed, main weights still transferring.
    #[test]
    fn test_the_state_a_restart_actually_left() {
        let root = weights_root();
        let d = model_dir(root.path(), &[MMPROJ, &format!("{WEIGHTS}.part")]);
        assert!(!gguf_exists(&d));
    }

    /// Weights done, mmproj still transferring — vision would be missing.
    #[test]
    fn test_the_other_ordering() {
        let root = weights_root();
        let d = model_dir(root.path(), &[WEIGHTS, &format!("{MMPROJ}.part")]);
        assert!(!gguf_exists(&d));
    }

    #[test]
    fn test_a_part_under_bin_counts_too() {
        let root = weights_root();
        let d = model_dir(root.path(), &[&format!("bin/{WEIGHTS}"), &format!("bin/{MMPROJ}.part")]);
        assert!(!gguf_exists(&d));
    }

    /// Complete weights in `bin/` do not excuse an unfinished sibling.
    #[test]
    fn test_a_part_anywhere_disqualifies_the_model() {
        let root = weights_root();
        let d = model_dir(root.path(), &[&format!("bin/{WEIGHTS}"), "something.gguf.part"]);
        assert!(!gguf_exists(&d));
    }
}

/// `TestAProjectorIsNotAModel` — `resolve_gguf_path` already refuses to load
/// one; so must this.
mod test_a_projector_is_not_a_model {
    use super::*;

    /// Parametrized over `[MMPROJ, "mmproj.gguf", "clip-vision.gguf"]`.
    #[test]
    fn test_a_companion_alone_is_not_servable() {
        for name in [MMPROJ, "mmproj.gguf", "clip-vision.gguf"] {
            let root = weights_root();
            let d = model_dir(root.path(), &[name]);
            assert!(!gguf_exists(&d), "{name} alone was taken for weights");
        }
    }

    #[test]
    fn test_but_it_does_not_disqualify_real_weights() {
        let root = weights_root();
        let d = model_dir(root.path(), &[WEIGHTS, MMPROJ]);
        assert!(gguf_exists(&d));
    }
}

/// `TestWhatMustStillBeAccepted` — the fix must not send a finished download
/// round again.
mod test_what_must_still_be_accepted {
    use super::*;

    #[test]
    fn test_weights_and_companion() {
        let root = weights_root();
        let d = model_dir(root.path(), &[WEIGHTS, MMPROJ]);
        assert!(gguf_exists(&d));
    }

    /// A model with no vision half is complete with one file.
    #[test]
    fn test_weights_alone() {
        let root = weights_root();
        let d = model_dir(root.path(), &[WEIGHTS]);
        assert!(gguf_exists(&d));
    }

    #[test]
    fn test_weights_under_bin() {
        let root = weights_root();
        let d = model_dir(root.path(), &[&format!("bin/{WEIGHTS}")]);
        assert!(gguf_exists(&d));
    }

    #[test]
    fn test_an_empty_directory() {
        let root = weights_root();
        let d = model_dir(root.path(), &[]);
        assert!(!gguf_exists(&d));
    }

    #[test]
    fn test_a_directory_that_is_not_there() {
        let root = weights_root();
        let missing = root.path().join("never").join("created");
        assert!(!gguf_exists(&missing.to_string_lossy()));
    }

    #[test]
    fn test_unrelated_files_are_not_weights() {
        let root = weights_root();
        let d = model_dir(root.path(), &["config.json", "tokenizer.model"]);
        assert!(!gguf_exists(&d));
    }
}
