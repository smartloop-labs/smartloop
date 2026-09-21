"""Tests for the in-flight download registry.

The registry is what stops the disk sweep deleting a download that is still
running, so a leak in either direction matters: a mark left behind strands the
space forever, and a mark dropped early is the data loss it exists to prevent.
"""

import os
import threading

import pytest

from smartloop.model.downloader import inflight
from smartloop.model.downloader.inflight import downloading, is_downloading


@pytest.fixture(autouse=True)
def _clean_registry():
    """No test may inherit another's marks, or a leak passes unnoticed."""
    inflight._active.clear()
    yield
    inflight._active.clear()


def test_marks_for_the_duration_of_the_block(tmp_path):
    target = str(tmp_path / "models" / "org" / "repo")

    assert not is_downloading(target)
    with downloading(target):
        assert is_downloading(target)
    assert not is_downloading(target)
    assert inflight._active == {}


def test_nesting_is_refcounted(tmp_path):
    """Two concurrent downloads of one model: the first to finish must not
    unmark it while the second is still writing."""
    target = str(tmp_path / "models" / "org" / "repo")

    with downloading(target):
        with downloading(target):
            assert is_downloading(target)
        assert is_downloading(target), "the inner exit released the outer mark"
    assert not is_downloading(target)


def test_an_exception_still_unmarks(tmp_path):
    """A failed download must not strand the directory as permanently live."""
    target = str(tmp_path / "models" / "org" / "repo")

    with pytest.raises(ConnectionError):
        with downloading(target):
            raise ConnectionError("transfer died")

    assert not is_downloading(target)
    assert inflight._active == {}


def test_a_symlinked_alias_is_the_same_directory(tmp_path):
    """The mark is written with the downloader's path and read with the
    sweep's, so the two have to canonicalise the same way.  This is not
    hypothetical on macOS, where /var resolves to /private/var."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(str(real), str(link))

    with downloading(str(real)):
        assert is_downloading(str(link))


def test_a_relative_alias_is_the_same_directory(tmp_path):
    target = tmp_path / "models" / "org" / "repo"
    target.mkdir(parents=True)
    alias = str(target / ".." / "repo")

    with downloading(str(target)):
        assert is_downloading(alias)


def test_containment_covers_a_bin_subdirectory(tmp_path):
    """Weights can land in <model_dir>/bin, so an exact match is not enough."""
    target = tmp_path / "models" / "org" / "repo"
    (target / "bin").mkdir(parents=True)

    with downloading(str(target)):
        assert is_downloading(str(target / "bin"))


def test_the_parent_is_covered_too(tmp_path):
    """Deleting models/<org> would destroy a download into <org>/<repo>, so the
    question "is it safe to remove this" has to answer no for the parent."""
    org = tmp_path / "models" / "org"
    target = org / "repo"
    target.mkdir(parents=True)

    with downloading(str(target)):
        assert is_downloading(str(org))


def test_a_sibling_is_not_covered(tmp_path):
    """Protection must stay narrow: one model downloading cannot strand the
    space of the model beside it."""
    org = tmp_path / "models" / "org"
    target = org / "repo"
    target.mkdir(parents=True)
    sibling = org / "other-repo"
    sibling.mkdir()

    with downloading(str(target)):
        assert not is_downloading(str(sibling))


def test_concurrent_marks_leave_nothing_behind(tmp_path):
    """The refcount's whole justification: two threads overlapping on one
    directory must leave it clean, not marked or negative."""
    target = str(tmp_path / "models" / "org" / "repo")
    start = threading.Barrier(2)
    seen = []

    def worker():
        start.wait()
        with downloading(target):
            seen.append(is_downloading(target))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == [True, True]
    assert inflight._active == {}
