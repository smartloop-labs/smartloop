"""An install that kept its state in a store sandbox, brought forward.

The framework home is ``~/.smartloop`` in every build, so the app and the CLI
share one workspace.  Builds that shipped before the home was resolved past
the sandbox put it under the rewritten ``$HOME`` instead, and those installs
still have their workspace, projects and model weights in there: the move has
to happen on their behalf, before anything reads the home, or an existing user
starts the new build to an empty workspace.
"""

import os

import pytest

from smartloop.workspace.migrations import ADOPTED_MARKER, adopt_confined_home

_CONTAINER = "Library/Containers/ai.smartloop.studio/Data"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A container-confined ``$HOME`` and the real home beside it."""
    monkeypatch.delenv("SLP_HOME", raising=False)
    monkeypatch.delenv("SNAP_NAME", raising=False)
    monkeypatch.delenv("SNAP_REAL_HOME", raising=False)

    real = tmp_path / "Users" / "tester"
    container = real / _CONTAINER
    old = container / ".smartloop"
    (old / "workspace-1" / "models").mkdir(parents=True)
    (old / "config.json").write_text('{"id": "workspace-1"}')
    (old / ".credentials").write_text("secret")
    (old / "workspace-1" / "models" / "weights.gguf").write_text("x" * 32)

    monkeypatch.setenv("HOME", str(container))
    return {"real": real, "old": old, "dest": real / ".smartloop"}


class TestBringingTheStateForward:
    def test_the_whole_home_moves_to_the_real_one(self, homes):
        assert adopt_confined_home(str(homes["dest"])) == str(homes["old"])

        dest = homes["dest"]
        assert (dest / "config.json").read_text() == '{"id": "workspace-1"}'
        assert (dest / ".credentials").read_text() == "secret"
        # Model weights run to gigabytes; they travel with everything else.
        assert (dest / "workspace-1" / "models" / "weights.gguf").exists()

    def test_the_old_home_is_emptied_and_marked(self, homes):
        adopt_confined_home(str(homes["dest"]))
        old = homes["old"]
        assert not (old / "config.json").exists()
        assert (old / ADOPTED_MARKER).read_text().strip() == str(homes["dest"])

    def test_a_second_start_does_nothing(self, homes):
        adopt_confined_home(str(homes["dest"]))
        # Whatever the running install has written since must not be walked
        # over by a migration that already happened.
        (homes["dest"] / "config.json").write_text('{"id": "moved-on"}')
        assert adopt_confined_home(str(homes["dest"])) is None
        assert (homes["dest"] / "config.json").read_text() == '{"id": "moved-on"}'


class TestWhenItDeclines:
    def test_an_unconfined_home_is_not_a_migration(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLP_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".smartloop").mkdir()
        assert adopt_confined_home(str(tmp_path / ".smartloop")) is None

    def test_an_explicit_slp_home_outranks_it(self, homes, monkeypatch):
        # A service unit, a test, or a packaged build that means to stay in
        # its container said where the home is. That instruction wins.
        monkeypatch.setenv("SLP_HOME", str(homes["old"]))
        assert adopt_confined_home(str(homes["dest"])) is None
        assert (homes["old"] / "config.json").exists()

    def test_a_workspace_already_at_the_real_home_is_left_alone(self, homes):
        # Two live workspaces: merging them would be a guess about whose
        # project list is the real one, so the user keeps both.
        homes["dest"].mkdir(parents=True)
        (homes["dest"] / "config.json").write_text('{"id": "the-cli-one"}')

        assert adopt_confined_home(str(homes["dest"])) is None
        assert (homes["dest"] / "config.json").read_text() == '{"id": "the-cli-one"}'
        assert (homes["old"] / "config.json").exists()

    def test_nothing_to_bring_forward(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLP_HOME", raising=False)
        container = tmp_path / "Users" / "tester" / _CONTAINER
        container.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(container))
        assert adopt_confined_home(str(tmp_path / "Users" / "tester" / ".smartloop")) is None


class TestASnapInstall:
    def test_the_revision_directory_is_brought_forward_too(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLP_HOME", raising=False)
        monkeypatch.delenv("SNAP_REAL_HOME", raising=False)
        real = tmp_path / "home" / "tester"
        confined = real / "snap" / "smartloop" / "42"
        old = confined / ".smartloop"
        old.mkdir(parents=True)
        (old / "config.json").write_text('{"id": "workspace-1"}')

        monkeypatch.setenv("HOME", str(confined))
        monkeypatch.setenv("SNAP_NAME", "smartloop")

        dest = real / ".smartloop"
        assert adopt_confined_home(str(dest)) == str(old)
        assert (dest / "config.json").read_text() == '{"id": "workspace-1"}'
