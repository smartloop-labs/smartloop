//! Port of `tests/conftest.py` (shared fixtures).

// ===== BEGIN unit: utils =====
// Port of the `_restore_root_logging` autouse fixture.

/// Snapshot of the root logger (handlers + level), restored on drop.
///
/// A test that points the root logger at a queue and leaves it installed
/// turns every later test's log call into a write nobody reads.  Keep such a
/// leak local to the test that made it:
///
/// ```ignore
/// let _root = crate::shared::restore_root_logging();
/// ```
pub struct RootLoggingGuard {
    handlers: Vec<smartloop::utils::logging_setup::RootHandler>,
    level: i32,
}

/// Take a [`RootLoggingGuard`] for the current test.
pub fn restore_root_logging() -> RootLoggingGuard {
    use smartloop::utils::logging_setup::{get_logger, root_handlers};
    RootLoggingGuard { handlers: root_handlers(), level: get_logger("").level() }
}

impl Drop for RootLoggingGuard {
    fn drop(&mut self) {
        use smartloop::utils::logging_setup::{get_logger, set_root_handlers};
        set_root_handlers(std::mem::take(&mut self.handlers));
        get_logger("").set_level(self.level);
    }
}
// ===== END unit: utils =====
