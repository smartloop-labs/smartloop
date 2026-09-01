"""Reasoning-block ("thinking") handling for models that emit one.

Hybrid-reasoning models answer in two parts: an internal monologue, then the
reply meant for the reader.  Only the reply may reach callers — everything
downstream of the LLM wrapper (the JSON planner, the DAG, retrieval, the chat
SSE stream) treats the generated text as the answer, so an unfiltered monologue
is displayed to the user and parsed as data.

Reasoning is requested per model by :meth:`SLPModel.thinking_default` (and per
request from the chat endpoint), which sets the ``enable_thinking`` variable the
model's own chat template reads; this module removes what it produces.

The delimiters are per-family, read off the templates in the GGUFs we ship:

* ``<think>…</think>`` — Qwen3.5 (``{%- if enable_thinking is defined and
  enable_thinking is true %}``).
* ``<|channel>thought…<channel|>`` — Gemma 4's thought channel (its template
  carries a ``strip_thinking`` macro that splits on exactly these).

Filtering runs unconditionally — a model that never emits the tags passes
through untouched — so a template that reasons without us asking is handled too.
A close tag ahead of any open tag is also handled, because a template can
pre-fill the opening marker into the generation prompt, leaving the model to
emit only the close.
"""

from __future__ import annotations

# (open, close) per model family.  Order matters only for readability; the
# filter always acts on whichever marker appears first in the text.
REASONING_BLOCKS = (
    ("<think>", "</think>"),
    ("<|channel>", "<channel|>"),
)

OPEN_TAGS = tuple(open_tag for open_tag, _ in REASONING_BLOCKS)
CLOSE_TAGS = tuple(close_tag for _, close_tag in REASONING_BLOCKS)
_CLOSE_FOR = dict(REASONING_BLOCKS)
_ALL_TAGS = OPEN_TAGS + CLOSE_TAGS


def _first_of(text: str, tags) -> tuple[int, str | None]:
    """Index and identity of the earliest of ``tags`` in ``text``."""
    best, found = -1, None
    for tag in tags:
        at = text.find(tag)
        if at != -1 and (best == -1 or at < best):
            best, found = at, tag
    return best, found


def _partial_tag_len(text: str, tags) -> int:
    """Length of the trailing run of ``text`` that could still become a tag.

    Streamed chunks split anywhere, so ``"<thi"`` at the end of a chunk may be
    the start of ``<think>`` — it has to be held back rather than emitted.
    """
    longest = max(len(t) for t in tags)
    for size in range(min(longest - 1, len(text)), 0, -1):
        tail = text[-size:]
        if any(t.startswith(tail) for t in tags):
            return size
    return 0


class ThinkingFilter:
    """Incremental filter that drops reasoning blocks from generated text.

    Feed it chunks in order; each :meth:`feed` returns the text that is safe to
    show.  Call :meth:`flush` once at the end to release anything held back
    (a trailing partial tag that turned out to be ordinary text).
    """

    def __init__(self) -> None:
        self._buf = ""
        self._awaiting: str | None = None   # close tag we are inside of
        self._thought: list[str] = []
        self._emitted = False
        self.saw_thinking = False

    @property
    def unterminated(self) -> bool:
        """True when generation stopped inside a reasoning block.

        Happens when ``max_tokens`` cuts the model off mid-thought; the reply
        it was reasoning toward was never produced.
        """
        return self._awaiting is not None

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []

        while self._buf:
            if self._awaiting is not None:
                at = self._buf.find(self._awaiting)
                if at == -1:
                    # Withhold the reasoning, keeping only what could still
                    # turn out to be a split closing tag.
                    hold = _partial_tag_len(self._buf, (self._awaiting,))
                    keep = len(self._buf) - hold
                    self._thought.append(self._buf[:keep])
                    self._buf = self._buf[keep:]
                    break
                self._thought.append(self._buf[:at])
                self._buf = self._buf[at + len(self._awaiting):]
                self._awaiting = None
                continue

            open_at, open_tag = _first_of(self._buf, OPEN_TAGS)
            close_at, close_tag = _first_of(self._buf, CLOSE_TAGS)
            # A close tag ahead of any open tag means the chat template
            # pre-filled the opening marker into the prompt: everything up to
            # it is reasoning, not answer.
            if close_at != -1 and (open_at == -1 or close_at < open_at):
                self._thought.append(self._buf[:close_at])
                self._buf = self._buf[close_at + len(close_tag):]
                self.saw_thinking = True
                continue
            if open_at == -1:
                hold = _partial_tag_len(self._buf, _ALL_TAGS)
                keep = len(self._buf) - hold
                out.append(self._buf[:keep])
                self._buf = self._buf[keep:]
                break
            out.append(self._buf[:open_at])
            self._buf = self._buf[open_at + len(open_tag):]
            self._awaiting = _CLOSE_FOR[open_tag]
            self.saw_thinking = True

        text = "".join(out)
        if text.strip():
            self._emitted = True
        return text

    def flush(self) -> str:
        """Release held-back text once generation has ended.

        When the model never closed its reasoning block and produced no reply,
        the withheld monologue is returned rather than nothing: a truncated
        thought is poor output, an empty string is a failed request.
        """
        if self._awaiting is not None:
            if self._emitted:
                return ""
            return "".join(self._thought).strip()
        tail, self._buf = self._buf, ""
        return tail


def strip_thinking(text: str) -> str:
    """Return ``text`` with any reasoning block removed."""
    if not text:
        return text
    f = ThinkingFilter()
    return (f.feed(text) + f.flush()).strip()
