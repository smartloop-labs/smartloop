"""Measure a client against the in-process path before switching the default.

The transport is the risky part of moving the model out of the agent: it sits
on the inference hot path, and a per-message overhead that costs more than the
token it delivers would show up as a slower answer, not as an error.  So the
default does not move on argument — it moves on a number from here.

Run against whatever this process would use::

    python -m smartloop.model_host.bench

or against a specific host, alongside the in-process baseline::

    python -m smartloop.model_host.bench --address ~/.smartloop/model-host.sock

What it reports, per client: wall-clock to the first chunk (the latency the
user sees before text starts moving), total wall-clock, chunks, characters,
and characters per second.  The comparison that matters is *time to first
chunk* and *characters per second* — a transport that loses on either is not
ready to be the default.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

from smartloop.model_host.client import (
    InProcessModelClient, ModelClient, RemoteModelClient,
)

DEFAULT_PROMPT = (
    "<system>You are a helpful assistant.</system>\n"
    "<user>Explain what a vector index is, in one paragraph.</user>"
)


class Timing:
    """One streamed generation, timed."""

    __slots__ = ("first_chunk_s", "total_s", "chunks", "chars")

    def __init__(self, first_chunk_s: float, total_s: float,
                 chunks: int, chars: int) -> None:
        self.first_chunk_s = first_chunk_s
        self.total_s = total_s
        self.chunks = chunks
        self.chars = chars

    @property
    def chars_per_second(self) -> float:
        return self.chars / self.total_s if self.total_s else 0.0


def time_one_stream(client: ModelClient, prompt: str) -> Timing:
    """Stream *prompt* once and time it."""
    started = time.perf_counter()
    first: Optional[float] = None
    chunks = chars = 0
    for text in client.stream(prompt):
        if first is None:
            first = time.perf_counter() - started
        chunks += 1
        chars += len(text)
    total = time.perf_counter() - started
    return Timing(first if first is not None else total, total, chunks, chars)


def benchmark(client: ModelClient, prompt: str = DEFAULT_PROMPT,
              runs: int = 3, warmup: bool = True) -> dict:
    """Stream *prompt* *runs* times and summarise.

    One warmup pass first: the first generation after a load pays for the
    prompt-cache miss and, for a remote client, for opening the connection.
    Including it would flatter or damn the transport for something that
    happens once.
    """
    if warmup:
        time_one_stream(client, prompt)

    timings = [time_one_stream(client, prompt) for _ in range(runs)]
    return {
        "client": type(client).__name__,
        "runs": runs,
        "first_chunk_s": statistics.median(t.first_chunk_s for t in timings),
        "total_s": statistics.median(t.total_s for t in timings),
        "chars": statistics.median(t.chars for t in timings),
        "chunks": statistics.median(t.chunks for t in timings),
        "chars_per_second": statistics.median(t.chars_per_second for t in timings),
    }


def format_report(results: list[dict]) -> str:
    """A fixed-width table, and the verdict the decision actually turns on."""
    header = (
        f"{'client':<24}{'first chunk':>13}{'total':>10}"
        f"{'chunks':>9}{'chars':>9}{'chars/s':>11}"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        lines.append(
            f"{row['client']:<24}{row['first_chunk_s']:>12.3f}s"
            f"{row['total_s']:>9.3f}s{row['chunks']:>9.0f}"
            f"{row['chars']:>9.0f}{row['chars_per_second']:>11.1f}"
        )

    if len(results) == 2:
        base, other = results
        lines.append("")
        lines.append(
            f"{other['client']} vs {base['client']}: "
            f"first chunk {_ratio(other['first_chunk_s'], base['first_chunk_s'])}, "
            f"throughput {_ratio(base['chars_per_second'], other['chars_per_second'])}"
        )
    return "\n".join(lines)


def _ratio(numerator: float, denominator: float) -> str:
    """``1.04x slower`` / ``0.98x faster`` — signed the way a reader expects."""
    if not denominator:
        return "n/a"
    factor = numerator / denominator
    return f"{factor:.2f}x {'slower' if factor >= 1 else 'faster'}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smartloop.model_host.bench",
        description="Compare a model host against the in-process path",
    )
    parser.add_argument("--address", help="model host to measure; omit to measure "
                                          "only the in-process path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-in-process", action="store_true",
                        help="measure only the host (no local model load)")
    args = parser.parse_args()

    results: list[dict] = []

    if not args.skip_in_process:
        from smartloop.api.services import get_model_service

        service = get_model_service()
        service.load_model()
        results.append(benchmark(InProcessModelClient(service), args.prompt, args.runs))

    if args.address:
        remote = RemoteModelClient(args.address)
        try:
            results.append(benchmark(remote, args.prompt, args.runs))
        finally:
            remote.close()

    print(format_report(results))


if __name__ == "__main__":
    main()
