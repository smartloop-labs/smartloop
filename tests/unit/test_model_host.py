"""A project agent reaches the generative model through a client, not directly.

The model is the one thing that cannot be per-project: same-device weight
copies serialize in the driver and each carries its own KV cache, so N projects
would mean N resident copies competing for one device. The split therefore runs
between project work and inference — these tests cover the seam and the socket
transport across it.
"""

import os
import threading
import time

import numpy as np
import pytest

from smartloop.model_host import protocol
from smartloop.model_host.client import (
    InProcessModelClient, ModelClient, ModelHostError, RemoteModelClient,
    default_socket_path, get_model_client, parse_address,
)
from smartloop.model_host.host import ModelHost, _unlink_stale_socket


class FakeDevice(ModelClient):
    """Stands in for the process that owns the GPU."""

    def __init__(self) -> None:
        self.calls: list = []
        self.chunks = ["Hello", ", ", "world"]
        self.chunk_delay = 0.0
        self.fail_with: Exception | None = None
        self.loaded = True
        self.yielded = 0

    def generate(self, prompt, model_name=None, capability=None,
                 max_tokens=None, thinking=None) -> str:
        self.calls.append(("generate", prompt, model_name, capability))
        self.options = {"max_tokens": max_tokens, "thinking": thinking}
        if self.fail_with:
            raise self.fail_with
        return f"answer to {prompt}"

    def stream(self, prompt, model_name=None, capability=None, cancelled=None,
               thinking=None):
        self.calls.append(("stream", prompt, model_name, capability))
        if self.fail_with:
            raise self.fail_with
        for chunk in self.chunks:
            if cancelled is not None and cancelled.is_set():
                return
            if self.chunk_delay:
                time.sleep(self.chunk_delay)
            self.yielded += 1
            yield chunk

    def encode_texts(self, texts, base_dir=None):
        self.calls.append(("encode_texts", tuple(texts), base_dir))
        if not texts:
            return None
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3)

    def is_loaded(self):
        return self.loaded

    def model_name(self):
        return "sl-mini"

    def model_label(self, capability=None):
        # Deliberately not a real model name: this asserts the label crosses
        # the wire intact, not what any particular model is called.
        return f"host-chosen-model{f' [{capability}]' if capability else ''}"

    def count_tokens(self, text):
        return len(text.split())

    def max_context_tokens(self):
        return 8192

    def response_max_context_tokens(self, capability=None):
        return 4096 if capability else 8192

    def has_capability_model(self, capability):
        return capability == "vision"

    def idle_seconds(self):
        return 12.5

    def load_model(self):
        self.calls.append(("load_model",))
        return "sl-mini"

    def unload_auxiliary_models(self, include_embedding=True):
        self.calls.append(("unload_auxiliary_models", include_embedding))

    def unload_embedding_model(self):
        self.calls.append(("unload_embedding_model",))


@pytest.fixture
def device() -> FakeDevice:
    return FakeDevice()


@pytest.fixture
def host(device, tmp_path):
    server = ModelHost(device, str(tmp_path / "model-host.sock"))
    server.start()
    yield server
    server.stop()


@pytest.fixture
def client(host):
    remote = RemoteModelClient(host.address, timeout=15)
    yield remote
    remote.close()


class TestFraming:
    def test_a_frame_round_trips(self, tmp_path):
        import socket

        left, right = socket.socketpair()
        try:
            protocol.send_frame(left, {"id": 1, "hello": "world"})
            assert protocol.recv_frame(right) == {"id": 1, "hello": "world"}
        finally:
            left.close()
            right.close()

    def test_generated_text_needs_no_escaping(self, tmp_path):
        """Why length-prefixed rather than newline-delimited."""
        import socket

        left, right = socket.socketpair()
        try:
            payload = {"text": "line one\nline two\n\nand \x00 a null"}
            protocol.send_frame(left, payload)
            assert protocol.recv_frame(right) == payload
        finally:
            left.close()
            right.close()

    def test_a_closed_peer_is_reported(self):
        import socket

        left, right = socket.socketpair()
        left.close()
        with pytest.raises(protocol.ConnectionClosed):
            protocol.recv_frame(right)
        right.close()

    def test_an_oversized_frame_is_refused(self):
        import socket

        left, right = socket.socketpair()
        try:
            left.sendall(protocol._HEADER.pack(protocol.MAX_FRAME_BYTES + 1))
            with pytest.raises(protocol.ProtocolError):
                protocol.recv_frame(right)
        finally:
            left.close()
            right.close()


class TestArrayTransport:
    def test_an_embedding_matrix_survives_the_wire(self):
        original = np.random.rand(4, 8).astype(np.float32)
        assert np.array_equal(
            protocol.decode_array(protocol.encode_array(original)), original
        )

    def test_none_passes_through(self):
        assert protocol.encode_array(None) is None
        assert protocol.decode_array(None) is None

    def test_a_non_contiguous_view_is_packed_correctly(self):
        original = np.random.rand(4, 8).astype(np.float32)[:, ::2]
        assert np.array_equal(
            protocol.decode_array(protocol.encode_array(original)), original
        )


class TestAddresses:
    @pytest.mark.parametrize("address,expected", [
        ("/tmp/model-host.sock", ("/tmp/model-host.sock", None)),
        ("unix:///tmp/a.sock", ("/tmp/a.sock", None)),
        ("unix:/tmp/a.sock", ("/tmp/a.sock", None)),
        ("127.0.0.1:9000", ("127.0.0.1", 9000)),
        (":9000", ("127.0.0.1", 9000)),
    ])
    def test_parsing(self, address, expected):
        assert parse_address(address) == expected

    def test_the_default_lives_in_the_home(self):
        assert default_socket_path("/home/x/.smartloop") == \
            "/home/x/.smartloop/model-host.sock"


class TestTheDefaultIsInProcess:
    def test_no_host_configured_means_no_transport(self, monkeypatch, device):
        monkeypatch.delenv("SLP_MODEL_HOST", raising=False)
        assert isinstance(get_model_client(device), InProcessModelClient)

    def test_a_configured_host_selects_the_socket_client(self, monkeypatch, device):
        monkeypatch.setenv("SLP_MODEL_HOST", "/tmp/somewhere.sock")
        client = get_model_client(device)
        assert isinstance(client, RemoteModelClient)
        assert client.address == "/tmp/somewhere.sock"

    def test_an_empty_setting_is_treated_as_unset(self, monkeypatch, device):
        monkeypatch.setenv("SLP_MODEL_HOST", "   ")
        assert isinstance(get_model_client(device), InProcessModelClient)


class TestTheInProcessClient:
    """It must call exactly what the caller used to call directly."""

    class _Service:
        def __init__(self):
            self.calls = []
            self._model_name = "sl-mini"

        def generate_completion(self, prompt, model_name=None, capability=None,
                                max_tokens=None, thinking=None):
            self.calls.append(("generate_completion", prompt, model_name, capability))
            self.options = {"max_tokens": max_tokens, "thinking": thinking}
            return "done"

        def stream_completion(self, prompt, model_name=None, cancelled=None,
                              capability=None, thinking=None):
            self.calls.append(("stream_completion", prompt, model_name, capability))

            class _Chunk:
                def __init__(self, text):
                    self.text = text

            return iter([_Chunk("a"), _Chunk("b")])

        def is_loaded(self):
            return True

        def count_tokens(self, text):
            return 7

        def get_max_context_tokens(self):
            return 8192

        def get_response_max_context_tokens(self, capability):
            return 4096

        def has_capability_model(self, capability):
            return True

        def response_model_label(self, capability=None):
            return "the-real-model (7B)"

        def idle_seconds(self):
            return 3.0

        def load_model(self):
            self.calls.append(("load_model",))

        def unload_auxiliary_models(self, include_embedding=True):
            self.calls.append(("unload_auxiliary_models", include_embedding))

        def unload_embedding_model(self):
            self.calls.append(("unload_embedding_model",))

    def test_generate_delegates(self):
        service = self._Service()
        assert InProcessModelClient(service).generate("hi", capability="vision") == "done"
        assert service.calls == [("generate_completion", "hi", None, "vision")]

    def test_stream_yields_plain_text(self):
        """The wire carries text, so both clients yield the same shape."""
        service = self._Service()
        assert list(InProcessModelClient(service).stream("hi")) == ["a", "b"]

    def test_the_questions_delegate(self):
        client = InProcessModelClient(self._Service())
        assert client.is_loaded() is True
        assert client.model_name() == "sl-mini"
        assert client.count_tokens("a b c") == 7
        assert client.max_context_tokens() == 8192
        assert client.response_max_context_tokens("vision") == 4096
        assert client.has_capability_model("vision") is True
        assert client.idle_seconds() == 3.0
        assert client.model_label() == "the-real-model (7B)"

    def test_load_reports_the_resident_model(self):
        service = self._Service()
        assert InProcessModelClient(service).load_model() == "sl-mini"
        assert ("load_model",) in service.calls


class TestTheSocketTransport:
    def test_generate(self, client, device):
        assert client.generate("who?", capability="general") == "answer to who?"
        assert ("generate", "who?", None, "general") in device.calls

    def test_stream(self, client, device):
        assert list(client.stream("tell me")) == ["Hello", ", ", "world"]

    def test_streamed_text_keeps_its_newlines(self, client, device):
        device.chunks = ["first\n", "\nsecond", '{"json": "like"}']
        assert list(client.stream("x")) == device.chunks

    def test_the_questions_cross_the_wire(self, client):
        assert client.is_loaded() is True
        assert client.model_name() == "sl-mini"
        assert client.model_label() == "host-chosen-model"
        assert client.model_label("vision") == "host-chosen-model [vision]"
        assert client.count_tokens("one two three") == 3
        assert client.max_context_tokens() == 8192
        assert client.response_max_context_tokens("vision") == 4096
        assert client.has_capability_model("vision") is True
        assert client.has_capability_model("coding") is False
        assert client.idle_seconds() == 12.5

    def test_embeddings_cross_the_wire_intact(self, client):
        result = client.encode_texts(["a", "b"])
        assert result.dtype == np.float32
        assert np.array_equal(result, np.arange(6, dtype=np.float32).reshape(2, 3))

    def test_an_unavailable_embedder_comes_back_as_none(self, client):
        assert client.encode_texts([]) is None

    def test_lifecycle_calls_reach_the_device(self, client, device):
        assert client.load_model() == "sl-mini"
        client.unload_auxiliary_models(include_embedding=False)
        client.unload_embedding_model()
        assert ("unload_auxiliary_models", False) in device.calls
        assert ("unload_embedding_model",) in device.calls

    def test_many_requests_share_one_connection(self, client, device):
        for _ in range(5):
            client.generate("hi")
        assert len([c for c in device.calls if c[0] == "generate"]) == 5


class TestFailures:
    def test_a_device_error_surfaces_at_the_caller(self, client, device):
        device.fail_with = ValueError("no model loaded")
        with pytest.raises(ModelHostError, match="no model loaded"):
            client.generate("hi")

    def test_the_connection_survives_a_failed_request(self, client, device):
        device.fail_with = ValueError("boom")
        with pytest.raises(ModelHostError):
            client.generate("hi")
        device.fail_with = None
        assert client.generate("again") == "answer to again"

    def test_an_unknown_operation_is_refused(self, client):
        with pytest.raises(ModelHostError, match="unknown operation"):
            client._call("teleport")

    def test_an_unreachable_host_is_reported_clearly(self, tmp_path):
        client = RemoteModelClient(str(tmp_path / "not-there.sock"), timeout=2)
        with pytest.raises(ModelHostError, match="unreachable"):
            client.is_loaded()

    def test_a_restarted_host_is_reconnected_to(self, device, tmp_path):
        address = str(tmp_path / "host.sock")
        first = ModelHost(device, address)
        first.start()
        client = RemoteModelClient(address, timeout=15)
        try:
            assert client.generate("before") == "answer to before"
            first.stop()

            with pytest.raises(ModelHostError):
                client.generate("during")

            second = ModelHost(device, address)
            second.start()
            try:
                assert client.generate("after") == "answer to after"
            finally:
                second.stop()
        finally:
            client.close()


class TestCancellation:
    def test_a_cancelled_stream_stops_early(self, client, device):
        device.chunks = [f"chunk-{i}" for i in range(50)]
        device.chunk_delay = 0.01
        cancelled = threading.Event()

        received = []
        for text in client.stream("long answer", cancelled=cancelled):
            received.append(text)
            if len(received) == 3:
                cancelled.set()

        # Both ends stop: the client stops receiving, and — the point of
        # sending CANCEL rather than just hanging up — the device stops being
        # asked for tokens it would compute for nobody.
        assert len(received) < 50
        assert device.yielded < 50

    def test_the_connection_is_usable_after_a_cancellation(self, client, device):
        device.chunks = [f"chunk-{i}" for i in range(50)]
        device.chunk_delay = 0.01
        cancelled = threading.Event()

        for index, _ in enumerate(client.stream("long", cancelled=cancelled)):
            if index == 2:
                cancelled.set()

        assert client.generate("next") == "answer to next"


class TestSelectionIsScopedToTheAskingProject:
    """Which model answers depends on the *requesting* project's settings.

    Per-project settings override the workspace defaults, so resolving against
    the host's own "current" project — the UI's selection — would answer one
    project's question with another project's chosen model.
    """

    def test_the_requesting_project_reaches_the_host(self, host, tmp_path):
        from smartloop.model_host.client import RemoteModelClient
        from smartloop.workspace.context import pin_project

        seen: list = []
        host.client.model_label = lambda capability=None: seen.append(
            __import__("smartloop.workspace.context", fromlist=["x"]).active_project_id()
        ) or "label"

        pin_project("beta")
        client = RemoteModelClient(host.address, timeout=15)
        try:
            client.model_label()
        finally:
            client.close()
            pin_project(None)

        assert seen == ["beta"]

    def test_two_projects_are_not_confused_for_each_other(self, host):
        from smartloop.model_host.client import RemoteModelClient
        from smartloop.workspace.context import active_project_id, pin_project

        host.client.model_label = lambda capability=None: active_project_id() or "none"

        answers = {}
        for project_id in ("alpha", "beta"):
            pin_project(project_id)
            client = RemoteModelClient(host.address, timeout=15)
            try:
                answers[project_id] = client.model_label()
            finally:
                client.close()
        pin_project(None)

        assert answers == {"alpha": "alpha", "beta": "beta"}

    def test_an_unpinned_caller_binds_nothing(self, host):
        """The supervisor's own calls keep resolving through its selection."""
        from smartloop.model_host.client import RemoteModelClient
        from smartloop.workspace.context import active_project_id, pin_project

        host.client.model_label = lambda capability=None: active_project_id() or "unbound"

        pin_project(None)
        client = RemoteModelClient(host.address, timeout=15)
        try:
            assert client.model_label() == "unbound"
        finally:
            client.close()


class TestConcurrency:
    def test_several_agents_share_one_host(self, host, device):
        """Three project agents, three connections, one device."""
        results: dict[int, str] = {}
        errors: list = []
        start = threading.Barrier(3, timeout=10)

        def agent(n: int) -> None:
            remote = RemoteModelClient(host.address, timeout=15)
            try:
                start.wait()
                results[n] = remote.generate(f"agent-{n}")
            except Exception as exc:  # pragma: no cover - failure detail
                errors.append(exc)
            finally:
                remote.close()

        threads = [threading.Thread(target=agent, args=(n,)) for n in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert not errors
        assert results == {n: f"answer to agent-{n}" for n in range(3)}

    def test_one_client_serializes_its_own_requests(self, client, device):
        """The exchange holds the connection, so frames never interleave."""
        errors: list = []
        results: list = []

        def call(n: int) -> None:
            try:
                results.append(client.generate(f"q{n}"))
            except Exception as exc:  # pragma: no cover - failure detail
                errors.append(exc)

        threads = [threading.Thread(target=call, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert not errors
        assert sorted(results) == sorted(f"answer to q{n}" for n in range(6))


class TestPlatformsWithoutUnixSockets:
    """Windows has no ``AF_UNIX``; the host has to still be reachable there.

    Feature-detected rather than guessed from ``sys.platform`` — which is what
    CPython's own ``socketpair`` does, and it means a build without AF_UNIX is
    handled wherever it turns up.
    """

    def test_the_default_address_becomes_loopback(self, monkeypatch):
        from smartloop.model_host import client as client_mod

        monkeypatch.setattr(client_mod, "HAS_UNIX_SOCKETS", False)

        assert client_mod.default_socket_path("/home/x/.smartloop") == "127.0.0.1:0"

    def test_loopback_binds_and_serves(self, device):
        """The fallback has to actually work, not just be selected."""
        server = ModelHost(device, "127.0.0.1:0")
        server.start()
        try:
            assert parse_address(server.address)[1] > 0
            client = RemoteModelClient(server.address, timeout=15)
            try:
                assert client.is_loaded() is True
                assert list(client.stream("hi")) == ["Hello", ", ", "world"]
            finally:
                client.close()
        finally:
            server.stop()

    def test_binding_a_path_without_af_unix_says_why(self, device, tmp_path):
        from smartloop.model_host import host as host_mod

        monkey = pytest.MonkeyPatch()
        monkey.setattr(host_mod, "HAS_UNIX_SOCKETS", False)
        try:
            with pytest.raises(OSError, match="no AF_UNIX"):
                ModelHost(device, str(tmp_path / "h.sock")).bind()
        finally:
            monkey.undo()

    def test_connecting_to_a_path_without_af_unix_says_why(self, monkeypatch, tmp_path):
        from smartloop.model_host import client as client_mod

        monkeypatch.setattr(client_mod, "HAS_UNIX_SOCKETS", False)

        client = client_mod.RemoteModelClient(str(tmp_path / "h.sock"), timeout=2)
        with pytest.raises(ModelHostError, match="cannot open"):
            client.is_loaded()

    def test_the_stale_socket_sweep_is_skipped(self, monkeypatch, tmp_path):
        """Nothing to unlink when the platform never made one."""
        from smartloop.model_host import host as host_mod

        monkeypatch.setattr(host_mod, "HAS_UNIX_SOCKETS", False)
        leftover = tmp_path / "stale.sock"
        leftover.write_text("")

        host_mod._unlink_stale_socket(str(leftover))  # must not raise

        assert leftover.exists()


class TestTheSocketFile:
    def test_it_is_not_world_readable(self, host):
        assert os.stat(host.address).st_mode & 0o777 == 0o600

    def test_it_is_removed_on_stop(self, device, tmp_path):
        server = ModelHost(device, str(tmp_path / "gone.sock"))
        server.start()
        assert os.path.exists(server.address)
        server.stop()
        assert not os.path.exists(server.address)

    def test_a_stale_socket_from_a_killed_host_is_cleared(self, device, tmp_path):
        path = tmp_path / "stale.sock"
        path.write_text("")  # a file where a socket used to be
        server = ModelHost(device, str(path))
        server.start()
        try:
            assert RemoteModelClient(server.address, timeout=15).is_loaded() is True
        finally:
            server.stop()

    def test_a_live_host_is_never_unlinked_from_under_its_clients(self, host):
        with pytest.raises(OSError, match="already listening"):
            _unlink_stale_socket(host.address)


class TestTheEmbedderIsSharedNotCopied:
    """BGE-M3 is ~417 MB; a child must not load its own alongside the host's."""

    @pytest.fixture(autouse=True)
    def _reset_embeddings(self, monkeypatch):
        from smartloop.db import embeddings

        monkeypatch.setattr(embeddings, "_remote", None)
        yield
        monkeypatch.setattr(embeddings, "_remote", None)

    @staticmethod
    def _as_a_child(monkeypatch, embeddings):
        """Stand where a per-project agent stands: no models in this process.

        The host fixture runs in-process and marks it the device owner, which
        is right for the host and wrong for the child being simulated.
        """
        from smartloop.model_host import client as client_mod

        monkeypatch.setattr(client_mod, "_device_owner", False)
        monkeypatch.setattr(
            embeddings, "_get_model",
            lambda *_a, **_k: pytest.fail("the child loaded its own embedder"),
        )

    def test_a_child_embeds_through_the_host(self, host, device, monkeypatch):
        from smartloop.db import embeddings

        monkeypatch.setenv("SLP_MODEL_HOST", host.address)
        self._as_a_child(monkeypatch, embeddings)

        result = embeddings.encode_texts(["a", "b"])

        assert np.array_equal(result, np.arange(6, dtype=np.float32).reshape(2, 3))
        assert any(c[0] == "encode_texts" for c in device.calls)

    def test_warmup_is_a_no_op_in_a_child(self, host, monkeypatch):
        from smartloop.db import embeddings

        monkeypatch.setenv("SLP_MODEL_HOST", host.address)
        self._as_a_child(monkeypatch, embeddings)

        embeddings.warmup("/tmp/ws")  # must not raise

    def test_the_host_itself_always_embeds_locally(self, host, monkeypatch):
        """It *is* the model; delegating would be a call to itself."""
        from smartloop.db import embeddings

        monkeypatch.setenv("SLP_MODEL_HOST", host.address)
        # ModelHost marked this process the device owner when it was built.
        assert embeddings._remote_embedder() is None

    def test_no_host_configured_embeds_locally(self, monkeypatch):
        from smartloop.db import embeddings

        monkeypatch.delenv("SLP_MODEL_HOST", raising=False)
        assert embeddings._remote_embedder() is None

    def test_an_unreachable_host_falls_back_rather_than_raising(
        self, monkeypatch, tmp_path
    ):
        """The documented contract: unavailable means None, and TF-IDF."""
        from smartloop.db import embeddings
        from smartloop.model_host import client as client_mod

        monkeypatch.setattr(client_mod, "_device_owner", False)
        monkeypatch.setenv("SLP_MODEL_HOST", str(tmp_path / "gone.sock"))
        assert embeddings.encode_texts(["a"]) is None


class TestTheBenchmarkReport:
    def test_it_times_a_stream(self, device):
        from smartloop.model_host.bench import benchmark

        device.chunks = ["one", "two", "three"]
        result = benchmark(device, "prompt", runs=2, warmup=False)

        assert result["client"] == "FakeDevice"
        assert result["chunks"] == 3
        assert result["chars"] == len("onetwothree")
        assert result["total_s"] >= 0

    def test_it_names_the_slower_transport(self):
        from smartloop.model_host.bench import format_report

        report = format_report([
            {"client": "InProcessModelClient", "runs": 3, "first_chunk_s": 0.10,
             "total_s": 1.0, "chars": 500, "chunks": 50, "chars_per_second": 500.0},
            {"client": "RemoteModelClient", "runs": 3, "first_chunk_s": 0.12,
             "total_s": 1.1, "chars": 500, "chunks": 50, "chars_per_second": 450.0},
        ])

        assert "1.20x slower" in report      # first chunk
        assert "1.11x slower" in report      # throughput
        assert "InProcessModelClient" in report
