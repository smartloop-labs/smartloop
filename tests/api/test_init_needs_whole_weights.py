"""``/v1/init`` must not call a half-downloaded model ready.

``_gguf_exists`` decides whether the route downloads at all, so whatever it
accepts is reported as available and never revisited.  It used to accept any
``.gguf``, which an interrupted multi-file download satisfies with the wrong
file: the companion that happened to finish first.

Observed live — a restart during the final verify left ``mmproj`` renamed and
the 2 GB weights as ``.part``.  Every later ``/v1/init`` answered "already
available", so the weights could never arrive; the workspace recorded the model
as downloaded and the client showed it ready.
"""

import pytest

from smartloop.api.routers.init import _gguf_exists

WEIGHTS = "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
MMPROJ = "mmproj-Ministral-3-3B-Instruct-2512.gguf"


@pytest.fixture
def weights_root(tmp_path_factory):
    """Somewhere to plant fake weights that is not the test's ``tmp_path``.

    ``no_real_downloads`` scans that tree afterwards for stray ``.gguf`` and
    ``.part`` files, on the reasoning that a download escaping the stub is
    otherwise silent.  Every case here plants exactly those, so the fixtures
    live where the guard is not looking — the arrangement
    ``test_init_promotes_a_downloaded_model`` uses for the same reason.
    """
    return tmp_path_factory.mktemp("weights")


def _model_dir(weights_root, *relative_names):
    d = weights_root / "google" / "some-model"
    for rel in relative_names:
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


class TestAnUnfinishedTransfer:
    """A ``.part`` is the downloader saying a file is not finished."""

    def test_the_state_a_restart_actually_left(self, weights_root):
        """mmproj renamed, main weights still transferring."""
        d = _model_dir(weights_root, MMPROJ, WEIGHTS + ".part")
        assert _gguf_exists(d) is False

    def test_the_other_ordering(self, weights_root):
        """Weights done, mmproj still transferring — vision would be missing."""
        d = _model_dir(weights_root, WEIGHTS, MMPROJ + ".part")
        assert _gguf_exists(d) is False

    def test_a_part_under_bin_counts_too(self, weights_root):
        d = _model_dir(weights_root, f"bin/{WEIGHTS}", f"bin/{MMPROJ}.part")
        assert _gguf_exists(d) is False

    def test_a_part_anywhere_disqualifies_the_model(self, weights_root):
        """Complete weights in ``bin/`` do not excuse an unfinished sibling."""
        d = _model_dir(weights_root, f"bin/{WEIGHTS}", "something.gguf.part")
        assert _gguf_exists(d) is False


class TestAProjectorIsNotAModel:
    """``resolve_gguf_path`` already refuses to load one; so must this."""

    @pytest.mark.parametrize("name", [MMPROJ, "mmproj.gguf", "clip-vision.gguf"])
    def test_a_companion_alone_is_not_servable(self, weights_root, name):
        d = _model_dir(weights_root, name)
        assert _gguf_exists(d) is False

    def test_but_it_does_not_disqualify_real_weights(self, weights_root):
        d = _model_dir(weights_root, WEIGHTS, MMPROJ)
        assert _gguf_exists(d) is True


class TestWhatMustStillBeAccepted:
    """The fix must not send a finished download round again."""

    def test_weights_and_companion(self, weights_root):
        d = _model_dir(weights_root, WEIGHTS, MMPROJ)
        assert _gguf_exists(d) is True

    def test_weights_alone(self, weights_root):
        """A model with no vision half is complete with one file."""
        d = _model_dir(weights_root, WEIGHTS)
        assert _gguf_exists(d) is True

    def test_weights_under_bin(self, weights_root):
        d = _model_dir(weights_root, f"bin/{WEIGHTS}")
        assert _gguf_exists(d) is True

    def test_an_empty_directory(self, weights_root):
        d = _model_dir(weights_root)
        assert _gguf_exists(d) is False

    def test_a_directory_that_is_not_there(self, weights_root):
        assert _gguf_exists(str(weights_root / "never" / "created")) is False

    def test_unrelated_files_are_not_weights(self, weights_root):
        d = _model_dir(weights_root, "config.json", "tokenizer.model")
        assert _gguf_exists(d) is False
