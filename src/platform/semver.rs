//! Port of `smartloop/platform/semver.py`.
//!
//! Minimal semver parsing/comparison for release update checks.
//!
//! Mirrors the ordering rules the cloud (slp-api) uses when picking the latest
//! published release: compare on (major, minor, patch), and a release WITHOUT a
//! prerelease sorts above the same version WITH one (1.0.15 > 1.0.15-beta.1).
//!
//! Kept intentionally small — no third-party dependency.

use once_cell::sync::Lazy;
use regex::Regex;

/// x.y.z with an optional -prerelease suffix (build metadata after '+' is ignored).
static SEMVER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")
        .expect("valid semver regex")
});

/// One prerelease identifier: numeric identifiers sort below alphanumeric ones
/// and compare numerically; alphanumeric ones compare lexically (semver §11).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrereleaseIdent {
    /// A purely numeric identifier (Python key `(0, int)`).
    Numeric(u64),
    /// An alphanumeric identifier (Python key `(1, str)`).
    Alpha(String),
}

/// Sort key for the prerelease portion.  A missing prerelease outranks any
/// present one (`Prerelease` < `Release` in the derived ordering).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrereleaseKey {
    /// Has a prerelease (Python key `(0, ids)`).
    Prerelease(Vec<PrereleaseIdent>),
    /// Released version (Python key `(1,)`).
    Release,
}

/// Sortable key for a parsed version: `(major, minor, patch, prerelease_key)`.
pub type VersionKey = (u64, u64, u64, PrereleaseKey);

/// Sort key for the prerelease portion (see [`PrereleaseKey`]).
fn prerelease_key(prerelease: Option<&str>) -> PrereleaseKey {
    todo!()
}

/// Return a sortable key for *version*, or `None` if it isn't valid semver.
pub fn parse(version: Option<&str>) -> Option<VersionKey> {
    todo!()
}

/// Return true when *candidate* is a strictly newer semver than *current*.
///
/// If either value can't be parsed, returns false (fail closed — don't claim
/// an update is available on garbage input).
pub fn is_newer(candidate: &str, current: &str) -> bool {
    todo!()
}
