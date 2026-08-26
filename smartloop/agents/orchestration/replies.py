"""Recognising degenerate or refusal replies from a small local model.

Both predicates guard the same failure: a reply that looks like content but
isn't, silently replacing the data a caller was waiting for.  Shared by the
document fan-out (where a bad extract would replace a document) and the
ANALYZE step (where it would replace the analysis).
"""


def is_none_extract(text: str) -> bool:
    """True when an extraction reply means "nothing relevant" (NONE).

    Matches an exact ``NONE`` (ignoring case, surrounding quotes and
    punctuation) or a short reply that starts with ``NONE`` (e.g.
    ``"NONE."`` / ``"NONE relevant"``).  Kept tight so a genuine extract
    that merely begins with the word "none" ("None of the chairs ...")
    is not mistaken for a refusal.
    """
    t = (text or "").strip().strip('."\'' "`").upper()
    if t == "NONE":
        return True
    return len(t) < 15 and t.startswith("NONE")


def is_placeholder_output(text: str) -> bool:
    """True when an LLM call returned a placeholder instead of real content.

    Backends substitute a sentinel when the model emits nothing (e.g.
    llama.cpp's ``"No response generated"`` empty-output fallback) or when
    generation fails outright (the ``"Error generating response: ..."``
    string).  Such text must never be surfaced as real output — an
    extracted document, an analysis summary, etc. — because it silently
    replaces the actual data the caller was waiting for.
    """
    t = (text or "").strip()
    low = t.lower()
    if low in {"no response generated", "no response"}:
        return True
    return low.startswith("error generating response") or low.startswith("error: ")
