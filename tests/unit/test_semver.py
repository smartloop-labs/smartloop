"""Tests for smartloop.platform.semver."""

from smartloop.platform import semver


class TestParse:
    def test_valid(self):
        assert semver.parse("1.2.3") is not None
        assert semver.parse("0.0.0") is not None
        assert semver.parse("1.0.15-beta.1") is not None
        assert semver.parse(" 1.2.3 ") is not None  # trimmed

    def test_invalid(self):
        assert semver.parse(None) is None
        assert semver.parse("") is None
        assert semver.parse("1.2") is None
        assert semver.parse("v1.2.3") is None
        assert semver.parse("garbage") is None


class TestIsNewer:
    def test_plain_ordering(self):
        assert semver.is_newer("1.0.15", "1.0.14") is True
        assert semver.is_newer("1.1.0", "1.0.99") is True
        assert semver.is_newer("2.0.0", "1.9.9") is True
        assert semver.is_newer("1.0.15", "1.0.15") is False
        assert semver.is_newer("1.0.14", "1.0.15") is False

    def test_prerelease_ordering(self):
        # a release outranks the same version with a prerelease
        assert semver.is_newer("1.0.15", "1.0.15-beta.1") is True
        assert semver.is_newer("1.0.15-beta.1", "1.0.15") is False
        # within prereleases, later identifier wins; numeric < alphanumeric per semver
        assert semver.is_newer("1.0.15-beta.2", "1.0.15-beta.1") is True
        assert semver.is_newer("1.0.0-rc.1", "1.0.0-beta.1") is True

    def test_fails_closed_on_bad_input(self):
        assert semver.is_newer("garbage", "1.0.0") is False
        assert semver.is_newer("1.0.0", "garbage") is False
        assert semver.is_newer("1.0.0", None) is False
