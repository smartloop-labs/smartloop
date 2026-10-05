//! Port of `tests/unit/test_semver.py`.

use smartloop::platform::semver;

mod test_parse {
    use super::*;

    #[test]
    fn test_valid() {
        assert!(semver::parse(Some("1.2.3")).is_some());
        assert!(semver::parse(Some("0.0.0")).is_some());
        assert!(semver::parse(Some("1.0.15-beta.1")).is_some());
        assert!(semver::parse(Some(" 1.2.3 ")).is_some()); // trimmed
    }

    #[test]
    fn test_invalid() {
        assert!(semver::parse(None).is_none());
        assert!(semver::parse(Some("")).is_none());
        assert!(semver::parse(Some("1.2")).is_none());
        assert!(semver::parse(Some("v1.2.3")).is_none());
        assert!(semver::parse(Some("garbage")).is_none());
    }
}

mod test_is_newer {
    use super::*;

    #[test]
    fn test_plain_ordering() {
        assert!(semver::is_newer("1.0.15", "1.0.14"));
        assert!(semver::is_newer("1.1.0", "1.0.99"));
        assert!(semver::is_newer("2.0.0", "1.9.9"));
        assert!(!semver::is_newer("1.0.15", "1.0.15"));
        assert!(!semver::is_newer("1.0.14", "1.0.15"));
    }

    #[test]
    fn test_prerelease_ordering() {
        // a release outranks the same version with a prerelease
        assert!(semver::is_newer("1.0.15", "1.0.15-beta.1"));
        assert!(!semver::is_newer("1.0.15-beta.1", "1.0.15"));
        // within prereleases, later identifier wins; numeric < alphanumeric per semver
        assert!(semver::is_newer("1.0.15-beta.2", "1.0.15-beta.1"));
        assert!(semver::is_newer("1.0.0-rc.1", "1.0.0-beta.1"));
    }

    #[test]
    fn test_fails_closed_on_bad_input() {
        assert!(!semver::is_newer("garbage", "1.0.0"));
        assert!(!semver::is_newer("1.0.0", "garbage"));
        // Python passes `None` as *current*; `is_newer` takes `&str`, so the
        // absent value is the empty string, which parses as nothing just the same.
        assert!(semver::parse(None).is_none());
        assert!(!semver::is_newer("1.0.0", ""));
    }
}
