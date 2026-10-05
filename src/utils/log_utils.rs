//! Port of `smartloop/utils/log_utils.py`.

use std::io::Write;

use console::Style;

use crate::constants::{LOGO, SLP_ACCENT};

/// Minimal logger interface standing in for Python's `logging.Logger` where a
/// logger object is passed around (e.g. `DocumentSplitter`, model factory).
pub trait LoggerLike: Send + Sync {
    fn info(&self, msg: &str);
    fn warning(&self, msg: &str);
    fn debug(&self, msg: &str);
    fn error(&self, msg: &str);
}

/// Parse a `#rrggbb` colour into its components.
fn hex_rgb(hex: &str) -> Option<(u8, u8, u8)> {
    let hex = hex.strip_prefix('#')?;
    if hex.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some((r, g, b))
}

fn try_print_logo(version: Option<&str>) -> std::io::Result<()> {
    let mut out = std::io::stdout().lock();
    let mut logo_style = Style::new().bold();
    if let Some((r, g, b)) = hex_rgb(SLP_ACCENT) {
        logo_style = logo_style.true_color(r, g, b);
    }
    writeln!(out, "{}", logo_style.apply_to(LOGO))?;

    if let Some(version) = version {
        writeln!(out, "{} {}", Style::new().dim().apply_to("Version:"), Style::new().green().apply_to(version))?;
    }

    writeln!(out)?;
    out.flush()
}

/// Print the Smartloop logo (bold, accent colour) and optional version line.
/// Falls back to plain text when the terminal can't render it.
pub fn print_logo(version: Option<&str>) {
    if try_print_logo(version).is_err() {
        // Fallback for terminals that don't support styled / Unicode output
        // (e.g. Windows).
        let suffix = version.map(|v| format!(" v{v}")).unwrap_or_default();
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "Smartloop Framework{suffix}");
        let _ = writeln!(out);
    }
}

/// Logger that prints coloured messages straight to the console
/// (info=blue, warning=yellow, debug=magenta, error=red).
#[derive(Debug, Clone, Default)]
pub struct ConsoleLogger;

impl ConsoleLogger {
    pub fn new() -> Self {
        ConsoleLogger
    }

    fn print(&self, style: Style, msg: &str) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{}", style.apply_to(msg));
    }
}

impl LoggerLike for ConsoleLogger {
    fn info(&self, msg: &str) {
        self.print(Style::new().blue(), msg);
    }

    fn warning(&self, msg: &str) {
        self.print(Style::new().yellow(), msg);
    }

    fn debug(&self, msg: &str) {
        self.print(Style::new().magenta(), msg);
    }

    fn error(&self, msg: &str) {
        self.print(Style::new().red(), msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_accent() {
        assert_eq!(hex_rgb("#ec4899"), Some((0xec, 0x48, 0x99)));
        assert_eq!(hex_rgb("ec4899"), None);
    }
}
