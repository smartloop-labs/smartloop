"""The user's home, read past the sandbox each store puts us in.

Both stores we ship through rewrite ``$HOME`` to a private per-app directory,
and a tilde expanded against it names a path the user has never put a file in:
on the Mac App Store build every local MCP server failed with an ENOENT under
``~/Library/Containers/<bundle-id>/Data``, and a strict snap would put the
framework home at ``~/snap/smartloop/<rev>/.smartloop`` instead of
``~/.smartloop``.  What neither rewrites is the passwd record, and snapd says
it outright in ``SNAP_REAL_HOME``.
"""

import os
import pwd

import pytest

from smartloop.paths import expand_home, real_home


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every case outside any confinement, whatever the runner is in."""
    monkeypatch.delenv("SNAP_REAL_HOME", raising=False)
    monkeypatch.delenv("SNAP_NAME", raising=False)


@pytest.fixture
def passwd_home():
    return pwd.getpwuid(os.getuid()).pw_dir


class TestRealHome:
    def test_an_ordinary_home_env_var_is_honoured(self, monkeypatch):
        # A service unit or a test that sets $HOME means it; only the two
        # confinement spellings are ever overridden.
        monkeypatch.setenv("HOME", "/home/tester")
        assert real_home() == "/home/tester"

    def test_an_app_sandbox_container_reads_through_to_passwd(
        self, monkeypatch, passwd_home,
    ):
        monkeypatch.setenv(
            "HOME", "/Users/tester/Library/Containers/ai.smartloop.studio/Data",
        )
        assert real_home() == passwd_home

    def test_snapd_is_taken_at_its_word(self, monkeypatch):
        # SNAP_REAL_HOME (snapd >= 2.46) is the answer stated outright, so
        # nothing has to be inferred from the shape of $HOME.
        monkeypatch.setenv("HOME", "/home/tester/snap/smartloop/42")
        monkeypatch.setenv("SNAP_NAME", "smartloop")
        monkeypatch.setenv("SNAP_REAL_HOME", "/home/tester")
        assert real_home() == "/home/tester"

    def test_an_older_snapd_falls_back_to_passwd(self, monkeypatch, passwd_home):
        # Before SNAP_REAL_HOME existed there is only $SNAP_USER_DATA, whose
        # shape says where the real home is — and passwd says it exactly.
        monkeypatch.setenv("HOME", "/home/tester/snap/smartloop/42")
        monkeypatch.setenv("SNAP_NAME", "smartloop")
        assert real_home() == passwd_home

    def test_a_snaps_environment_leaking_into_a_child_is_ignored(self, monkeypatch):
        # A snap that launches something else — a terminal emulator starting a
        # shell — hands its whole SNAP_* block to a child that is not confined
        # at all.  $HOME is what says whether we are the confined process, so
        # an inherited SNAP_REAL_HOME must not override a home that is already
        # real (and, in a test, already steered).
        monkeypatch.setenv("HOME", "/home/tester")
        monkeypatch.setenv("SNAP_NAME", "alacritty")
        monkeypatch.setenv("SNAP_REAL_HOME", "/home/someone-else")
        assert real_home() == "/home/tester"

    def test_a_snap_shaped_home_outside_a_snap_is_left_alone(self, monkeypatch):
        # ``~/snap/x/y`` is a perfectly ordinary directory to have as a home.
        # Without SNAP_NAME we are not in a snap and $HOME is not a rewrite.
        monkeypatch.setenv("HOME", "/home/tester/snap/smartloop/42")
        assert real_home() == "/home/tester/snap/smartloop/42"


class TestExpandHome:
    def test_a_tilde_expands_against_the_real_home(self, monkeypatch):
        monkeypatch.setattr("smartloop.paths.real_home", lambda: "/Users/tester")
        assert expand_home("~/.smartloop") == "/Users/tester/.smartloop"
        assert expand_home("~") == "/Users/tester"

    def test_an_absolute_path_is_returned_unchanged(self):
        assert expand_home("/opt/smartloop") == "/opt/smartloop"

    def test_another_users_home_is_left_to_expanduser(self, monkeypatch):
        # ``~root/...`` names somebody else's home, which only the passwd
        # lookup behind expanduser knows how to resolve.
        monkeypatch.setattr("smartloop.paths.real_home", lambda: "/Users/tester")
        assert expand_home("~root/x") == os.path.expanduser("~root/x")
