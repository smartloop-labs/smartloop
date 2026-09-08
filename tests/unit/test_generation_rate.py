"""How fast generation actually was, and what decided it.

"Slow" is not actionable and neither is a wall-clock total: a long prompt on a
fast model and a short prompt on a model that fell back to CPU take the same
ten seconds and want opposite fixes.  Prefill (time to the first token) and
decode rate separate them, and the streaming path — the one a user watches —
measured neither.

The load-time knobs behind both were fixed at 512 and 4 with no way to see or
change them; they are chosen from the device now, and logged.
"""

import logging
import time

from smartloop.constants import OPENING_TOKENS
from smartloop.llms.llama_cpp import LlamaCpp
from smartloop.llms.llm_factory import LlmFactory


class TestTheRateLine:
    def test_it_separates_prefill_from_decode(self):
        start = time.time() - 5.0        # generation began 5s ago
        first = time.time() - 4.0        # first token landed 1s in

        line = LlamaCpp._rate(3200, 48, start, first)

        assert "3200 prompt" in line
        assert "ttft 1.0" in line
        # 48 tokens over the 4s since the first one.
        assert "12.0 tok/s" in line

    def test_a_generation_that_produced_nothing_says_nothing(self):
        assert LlamaCpp._rate(3200, 0, time.time(), None) == ""
        assert LlamaCpp._rate(3200, 5, time.time(), None) == ""

    def test_a_single_token_does_not_divide_by_zero(self):
        now = time.time()
        assert "tok/s" in LlamaCpp._rate(10, 1, now, now)

    def test_the_opening_is_reported_apart_from_the_rest(self):
        """An average hides the ramp, and the ramp is what a reader notices."""
        now = time.time()
        start = now - 11.0
        first = now - 10.0               # 10s of decode
        opening = now - 6.0              # the first 32 tokens took 4s of it

        line = LlamaCpp._rate(900, 132, start, first, opening)

        # 132/10s overall, 32 in 4s at the start, 100 in the 6s after.
        assert "13.2 tok/s" in line
        assert f"first {OPENING_TOKENS}: 8.0" in line
        assert "then 16.7" in line

    def test_a_generation_too_short_to_have_two_halves_reports_one_rate(self):
        now = time.time()
        line = LlamaCpp._rate(900, OPENING_TOKENS, now - 3.0, now - 2.0, now - 1.0)
        assert "tok/s" in line
        assert "first" not in line


class TestBatchAndThreads:
    def _pick(self, n_gpu_layers, monkeypatch, **env):
        for key in ("SLP_LLM_N_BATCH", "SLP_LLM_N_THREADS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return LlmFactory._batch_and_threads(n_gpu_layers, logging.getLogger("t"))

    def test_a_gpu_load_gets_the_bigger_prefill_batch(self, monkeypatch):
        n_batch, _threads = self._pick(-1, monkeypatch)
        assert n_batch == 1024

    def test_a_cpu_load_gets_the_machine_s_threads(self, monkeypatch):
        """Only the layers that are not offloaded run on them, so a CPU-only
        load is exactly where a flat 4 left the machine idle."""
        import os

        _batch, n_threads = self._pick(0, monkeypatch)
        assert n_threads == (os.cpu_count() or 4)

    def test_the_environment_wins(self, monkeypatch):
        n_batch, n_threads = self._pick(
            -1, monkeypatch, SLP_LLM_N_BATCH="256", SLP_LLM_N_THREADS="2",
        )
        assert (n_batch, n_threads) == (256, 2)
