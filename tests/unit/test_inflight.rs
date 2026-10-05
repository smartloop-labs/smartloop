//! Port of `tests/unit/test_inflight.py`.
//!
//! Tests for the in-flight download registry.
//!
//! The registry is what stops the disk sweep deleting a download that is still
//! running, so a leak in either direction matters: a mark left behind strands
//! the space forever, and a mark dropped early is the data loss it exists to
//! prevent.

use std::path::Path;
use std::sync::{Arc, Barrier};

use serial_test::serial;
use smartloop::model::downloader::inflight::{self, downloading, is_downloading};

/// No test may inherit another's marks, or a leak passes unnoticed
/// (the `_clean_registry` autouse fixture).
struct CleanRegistry;

impl CleanRegistry {
    fn new() -> Self {
        inflight::reset_for_tests();
        CleanRegistry
    }
}

impl Drop for CleanRegistry {
    fn drop(&mut self) {
        inflight::reset_for_tests();
    }
}

fn s(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[test]
#[serial]
fn test_marks_for_the_duration_of_the_block() {
    let _clean = CleanRegistry::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = s(&tmp.path().join("models").join("org").join("repo"));

    assert!(!is_downloading(&target));
    {
        let _mark = downloading(&target);
        assert!(is_downloading(&target));
    }
    assert!(!is_downloading(&target));
    assert!(inflight::active_snapshot().is_empty());
}

/// Two concurrent downloads of one model: the first to finish must not unmark
/// it while the second is still writing.
#[test]
#[serial]
fn test_nesting_is_refcounted() {
    let _clean = CleanRegistry::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = s(&tmp.path().join("models").join("org").join("repo"));

    {
        let _outer = downloading(&target);
        {
            let _inner = downloading(&target);
            assert!(is_downloading(&target));
        }
        assert!(is_downloading(&target), "the inner exit released the outer mark");
    }
    assert!(!is_downloading(&target));
}

/// A failed download must not strand the directory as permanently live.
#[test]
#[serial]
fn test_an_exception_still_unmarks() {
    let _clean = CleanRegistry::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = s(&tmp.path().join("models").join("org").join("repo"));

    let result: Result<(), std::io::Error> = (|| {
        let _mark = downloading(&target);
        Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "transfer died"))
    })();
    let err = result.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::ConnectionReset);

    // A panic unwinding through the block releases the mark too.
    let target_for_panic = target.clone();
    let panicked = std::panic::catch_unwind(move || {
        let _mark = downloading(&target_for_panic);
        panic!("transfer died");
    });
    assert!(panicked.is_err());

    assert!(!is_downloading(&target));
    assert!(inflight::active_snapshot().is_empty());
}

/// The mark is written with the downloader's path and read with the sweep's,
/// so the two have to canonicalise the same way.  This is not hypothetical on
/// macOS, where /var resolves to /private/var.
#[cfg(unix)]
#[test]
#[serial]
fn test_a_symlinked_alias_is_the_same_directory() {
    let _clean = CleanRegistry::new();
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().join("real");
    std::fs::create_dir(&real).unwrap();
    let link = tmp.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let _mark = downloading(s(&real));
    assert!(is_downloading(s(&link)));
}

#[test]
#[serial]
fn test_a_relative_alias_is_the_same_directory() {
    let _clean = CleanRegistry::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("models").join("org").join("repo");
    std::fs::create_dir_all(&target).unwrap();
    let alias = s(&target.join("..").join("repo"));

    let _mark = downloading(s(&target));
    assert!(is_downloading(&alias));
}

/// Weights can land in <model_dir>/bin, so an exact match is not enough.
#[test]
#[serial]
fn test_containment_covers_a_bin_subdirectory() {
    let _clean = CleanRegistry::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("models").join("org").join("repo");
    std::fs::create_dir_all(target.join("bin")).unwrap();

    let _mark = downloading(s(&target));
    assert!(is_downloading(s(&target.join("bin"))));
}

/// Deleting models/<org> would destroy a download into <org>/<repo>, so the
/// question "is it safe to remove this" has to answer no for the parent.
#[test]
#[serial]
fn test_the_parent_is_covered_too() {
    let _clean = CleanRegistry::new();
    let tmp = tempfile::tempdir().unwrap();
    let org = tmp.path().join("models").join("org");
    let target = org.join("repo");
    std::fs::create_dir_all(&target).unwrap();

    let _mark = downloading(s(&target));
    assert!(is_downloading(s(&org)));
}

/// Protection must stay narrow: one model downloading cannot strand the space
/// of the model beside it.
#[test]
#[serial]
fn test_a_sibling_is_not_covered() {
    let _clean = CleanRegistry::new();
    let tmp = tempfile::tempdir().unwrap();
    let org = tmp.path().join("models").join("org");
    let target = org.join("repo");
    std::fs::create_dir_all(&target).unwrap();
    let sibling = org.join("other-repo");
    std::fs::create_dir(&sibling).unwrap();

    let _mark = downloading(s(&target));
    assert!(!is_downloading(s(&sibling)));
}

/// The refcount's whole justification: two threads overlapping on one
/// directory must leave it clean, not marked or negative.
#[test]
#[serial]
fn test_concurrent_marks_leave_nothing_behind() {
    let _clean = CleanRegistry::new();
    let tmp = tempfile::tempdir().unwrap();
    let target = s(&tmp.path().join("models").join("org").join("repo"));
    let start = Arc::new(Barrier::new(2));
    let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));

    let threads: Vec<_> = (0..2)
        .map(|_| {
            let start = Arc::clone(&start);
            let seen = Arc::clone(&seen);
            let target = target.clone();
            std::thread::spawn(move || {
                start.wait();
                let _mark = downloading(&target);
                seen.lock().push(is_downloading(&target));
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(*seen.lock(), vec![true, true]);
    assert!(inflight::active_snapshot().is_empty());
}
