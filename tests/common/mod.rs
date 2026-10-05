//! Port of `tests/conftest.py` (shared fixtures).

// ============================================================================
// api_core — `_reset_project_pin`.
// ============================================================================
#[allow(dead_code)]
pub mod api_core {
    use smartloop::workspace::context::pin_project;

    /// Unpin the process around a test (`_reset_project_pin`).
    ///
    /// The pin is process-wide by design — an agent child sets it once at
    /// spawn — so a test that pins and does not clean up leaks into every test
    /// that runs after it.  That failure is quiet in the worst way: a leaked
    /// pin makes the workspace-owner checks answer false, so assertions about
    /// "did not do the owner's work" keep passing for the wrong reason.
    pub struct ProjectPinReset;

    impl ProjectPinReset {
        pub fn new() -> Self {
            pin_project(None);
            ProjectPinReset
        }
    }

    impl Default for ProjectPinReset {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for ProjectPinReset {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                pin_project(None);
            } else {
                let _ = std::panic::catch_unwind(|| pin_project(None));
            }
        }
    }
}
// ============================================================================
// end api_core
// ============================================================================
