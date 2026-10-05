//! Port of `smartloop/utils/log_utils.py`.

use crate::constants::{LOGO, SLP_ACCENT};

/// Minimal logger interface standing in for Python's `logging.Logger` where a
/// logger object is passed around (e.g. `DocumentSplitter`, model factory).
pub trait LoggerLike: Send + Sync {
    fn info(&self, msg: &str);
    fn warning(&self, msg: &str);
    fn debug(&self, msg: &str);
    fn error(&self, msg: &str);
}

/// Print the Smartloop logo (bold, accent colour) and optional version line.
/// Falls back to plain text when the terminal can't render it.
pub fn print_logo(version: Option<&str>) {
    let _ = (version, LOGO, SLP_ACCENT);
    todo!()
}

/// Logger that prints coloured messages straight to the console
/// (info=blue, warning=yellow, debug=magenta, error=red).
#[derive(Debug, Clone, Default)]
pub struct ConsoleLogger;

impl ConsoleLogger {
    pub fn new() -> Self {
        ConsoleLogger
    }
}

impl LoggerLike for ConsoleLogger {
    fn info(&self, msg: &str) {
        let _ = msg;
        todo!()
    }

    fn warning(&self, msg: &str) {
        let _ = msg;
        todo!()
    }

    fn debug(&self, msg: &str) {
        let _ = msg;
        todo!()
    }

    fn error(&self, msg: &str) {
        let _ = msg;
        todo!()
    }
}
