"""Store handles are cached per project, not one slot for the whole process.

With the project bound per request, two chats hold two different projects at
once. A single cache slot would have them evicting each other's store on every
turn — reopening SQLite each time, and in the window between the check and the
rebuild, answering one project's query from the other's index.
"""

import threading

from smartloop.api.services.model_service import ModelService, _ProjectCaches


def _service() -> ModelService:
    svc = ModelService.__new__(ModelService)
    svc._project_caches = {}
    svc._project_caches_lock = threading.Lock()
    return svc


class TestOneSlotPerProject:
    def test_each_project_gets_its_own_slot(self):
        svc = _service()
        assert svc._caches("alpha") is not svc._caches("beta")

    def test_the_same_project_gets_the_same_slot(self):
        svc = _service()
        assert svc._caches("alpha") is svc._caches("alpha")

    def test_a_slot_starts_empty(self):
        cache = _service()._caches("alpha")
        assert cache.doc_store is None
        assert cache.workspace_indexer is None
        assert cache.feedback_indexer is None

    def test_a_handle_stored_for_one_project_is_invisible_to_another(self):
        svc = _service()
        svc._caches("alpha").doc_store = "alpha-store"
        assert svc._caches("beta").doc_store is None
        assert svc._caches("alpha").doc_store == "alpha-store"

    def test_concurrent_first_touches_agree_on_one_slot(self):
        """Two requests opening the same project at once must not build two."""
        svc = _service()
        start = threading.Barrier(8, timeout=10)
        seen: list = []
        lock = threading.Lock()

        def touch() -> None:
            start.wait()
            cache = svc._caches("alpha")
            with lock:
                seen.append(cache)

        threads = [threading.Thread(target=touch) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert len(seen) == 8
        assert all(c is seen[0] for c in seen)


class TestDroppingSlots:
    def test_dropping_one_project_leaves_the_others_alone(self):
        svc = _service()
        svc._caches("alpha").doc_store = "alpha-store"
        svc._caches("beta").doc_store = "beta-store"

        svc._drop_caches("alpha")

        assert svc._caches("alpha").doc_store is None
        assert svc._caches("beta").doc_store == "beta-store"

    def test_dropping_everything(self):
        svc = _service()
        svc._caches("alpha").doc_store = "alpha-store"
        svc._caches("beta").doc_store = "beta-store"

        svc._drop_caches()

        assert svc._caches("alpha").doc_store is None
        assert svc._caches("beta").doc_store is None

    def test_dropping_a_project_with_no_slot_is_not_an_error(self):
        _service()._drop_caches("never-touched")


class TestTheSlotItself:
    def test_it_carries_no_per_instance_dict(self):
        """__slots__: one of these exists per open project, so keep it small."""
        assert not hasattr(_ProjectCaches(), "__dict__")
