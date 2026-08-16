"""Web search agent — pulls the latest information/events from the web.

Web search is a Pro/premium feature. When enabled on the project, this agent
calls the hosted Smartloop endpoint ``GET {slp_api_base}/v1/websearch`` (which
returns a JSON array of result links), then fetches up to ``num_candidates``
links (via the FetchAgent, HTML/PDF → markdown), LLM-reranks them by relevance
to the query, and returns the top ``top_k`` sources so the LLM can reason over
up-to-date information.

Enablement is per project (``Project.web_search_enabled``), toggled via
``PATCH /v1/projects/{id}/websearch``. The hosted endpoint is itself gated by
the caller's plan — a 401/403 means the account isn't on a plan that includes
web search. Web search is part of explore: any failure (disabled, 401/403,
no credentials, request error, no results) is a silent skip that returns ``""``
so the model simply falls back to its own knowledge — no error or upsell is
injected into the answer.
"""

import logging
import re
from typing import List, Optional

from smartloop.search.providers import (
    NoSearchCredentials,
    SearchAuthError,
)

logger = logging.getLogger(__name__)

_WEBSEARCH_PATH = "/v1/websearch"

# Leading conversational filler stripped before the search query hits the
# hosted endpoint.  A raw message like "give me a list latest canyon bikes"
# otherwise matches on the filler word ("give") and returns dictionary pages
# instead of the actual subject.
_WEBSEARCH_FILLERS = (
    "give me a list of ",
    "give me a list ",
    "give a list of ",
    "give me ",
    "can you give me ",
    "could you give me ",
    "can you tell me ",
    "could you tell me ",
    "tell me about ",
    "tell me ",
    "show me ",
    "can you show me ",
    "can you ",
    "could you ",
    "i want to know about ",
    "i want to know ",
    "i want to ",
    "i would like to know ",
    "i would like ",
    "i'd like to know ",
    "i'd like ",
    "do you know about ",
    "do you know ",
    "do you have any info on ",
    "do you have ",
    "what are the best ",
    "what are the latest ",
    "what are ",
    "what is the latest ",
    "what is ",
    "whats the latest ",
    "whats ",
    "any info on ",
    "info on ",
    "about ",
    "a list of ",
    "list of ",
    "please ",
)

_WEBSEARCH_TRAILERS = (" please", " please?", " thanks", " thank you")

# Generic / temporal tokens that add no retrieval signal to a web query.
# The hosted endpoint otherwise latches onto them: "current stock prices for
# apple" returns current.com pages, not AAPL quotes.  Dropped across the WHOLE
# query (not just the leading filler).  A single-subject query like "electric
# current" is kept intact via the >= 2 remaining words floor below.
_WEBSEARCH_STOPWORDS = frozenset({
    "the", "a", "an", "for", "of", "to", "and", "or", "in", "on", "at",
    "with", "current", "latest", "today", "todays", "now", "please",
})


def normalize_search_query(raw: str) -> str:
    """Distill a conversational message into a web-search query.

    ``"give me a list latest canyon bikes"`` → ``"canyon bikes"``.
    Polite framing and filler is stripped from the front and trailing
    particles trimmed, then weak/temporal stopwords ("the", "for", "current",
    "latest") are dropped anywhere so the endpoint matches the actual subject
    instead of a filler word.  Falls back to the raw message when nothing
    meaningful remains.
    """
    if not raw:
        return raw
    q = re.sub(r"\s+", " ", raw.strip()).strip("?.,;:!").strip()
    # Strip leading filler iteratively — "please tell me about the weather"
    # sheds "please", then "tell me", then "about".
    for _ in range(6):
        low = " " + q.lower() + " "
        hit = next((f for f in _WEBSEARCH_FILLERS if low.startswith(" " + f)), None)
        if not hit:
            break
        q = q[len(hit):]
    q = re.sub(r"\s+", " ", q.strip()).strip("?.,;:!").strip()
    for tail in _WEBSEARCH_TRAILERS:
        if q.lower().endswith(tail):
            q = q[: -len(tail)].strip("?.,;:!").strip()
            break
    words = [w for w in q.split() if w.lower() not in _WEBSEARCH_STOPWORDS]
    if len(words) >= 2:
        q = " ".join(words)
    return q or raw.strip()


class WebSearchAgent:
    """Search the web via the hosted endpoint and return fetched content."""

    def __init__(
        self,
        fetch_agent=None,
        top_k: int = 3,
        num_candidates: int = 10,
        llm=None,
    ):
        # ``top_k`` links are returned; ``num_candidates`` is how many result
        # links we request and fetch so we can survive forbidden/JS-only pages
        # (some links 401/403 or need JavaScript — we skip them) and then
        # LLM-rerank the survivors by relevance.  ``llm`` (optional) judges the
        # rerank; without one the endpoint's positional order is kept.
        self._fetch_agent = fetch_agent
        self._top_k = top_k
        self._num_candidates = num_candidates
        self._llm = llm
        # URLs of the sources kept after the last search() (top_k, reranked
        # order).  Read by the orchestrator so the sources surface as links in
        # the final answer rather than only as context the model may not echo.
        self.last_sources: List[str] = []

    @property
    def fetch_agent(self):
        if self._fetch_agent is None:
            from smartloop.agents.fetch_agent import FetchAgent
            self._fetch_agent = FetchAgent()
        return self._fetch_agent

    @property
    def llm(self):
        return self._llm

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        num_candidates: Optional[int] = None,
    ) -> str:
        """Search the web for *query* and return fetched result content.

        Returns ``""`` (never raises, never a notice) when web search is
        disabled, forbidden (401/403 — plan gate), unauthenticated, or fails —
        so it silently skips and the model falls back to its own knowledge
        instead of surfacing an error/upsell in the answer.
        """
        if not self.is_enabled():
            logger.info("[web_search] disabled for this project — skipping")
            return ""

        want = top_k or self._top_k
        n = num_candidates or self._num_candidates
        self.last_sources = []

        # Search on the distilled subject, not the raw conversational message
        # ("give me a list latest canyon bikes" → "latest canyon bikes").
        query = normalize_search_query(query)

        try:
            links = self._fetch_links(query, n)
        except SearchAuthError:
            logger.info("[web_search] provider rejected credentials (401/403) — skipping")
            return ""
        except NoSearchCredentials:
            logger.info("[web_search] no credentials available — skipping")
            return ""
        except Exception as e:
            logger.warning("[web_search] request failed, skipping: %s", e)
            return ""

        if not links:
            logger.info("[web_search] no results for %r — skipping", query)
            return ""

        return self._fetch_top(query, links, want, n)

    # ---------------------------------------------------------------- endpoint

    def _fetch_links(self, query: str, count: int) -> List[str]:
        """Return result URLs for *query* from the configured search provider.

        Uses the user-registered provider (e.g. Brave) when one is configured,
        otherwise the hosted ``/v1/websearch`` endpoint.  Raises
        ``SearchAuthError`` when the provider rejects the credentials, and
        ``NoSearchCredentials`` when none are available.
        """
        from smartloop.search.providers import resolve_provider

        provider = resolve_provider()
        logger.info("[web_search] provider=%r for %r", provider.name, query)
        return provider.search(query, count)

    # ---------------------------------------------------------------- fetching

    def _fetch_top(
        self, query: str, links: List[str], top_k: int, num_candidates: int
    ) -> str:
        """Fetch the links, then LLM-rerank and keep the ``top_k`` most relevant.

        Fetches up to ``num_candidates`` usable sources (skipping forbidden /
        JS-only pages), re-orders them by LLM-judged relevance to *query*, and
        returns the top ``top_k``.  Falls back to the endpoint's positional
        order when no LLM is wired in or the rerank call fails.
        """
        fetched: List[tuple] = []   # (url, content)
        skipped: List[str] = []
        budget = max(num_candidates, top_k)
        for url in links:
            if len(fetched) >= budget:
                break
            content = ""
            try:
                content = self.fetch_agent.fetch(url, max_length=4000)
            except Exception as e:
                logger.info("[web_search] fetch failed for %s: %s", url, e)
            if self._is_usable(content):
                fetched.append((url, content.strip()))
            else:
                skipped.append(url)
                logger.info("[web_search] skipping unusable/forbidden link: %s", url)

        if not fetched:
            # Nothing fetched cleanly — fall back to the bare link list so the
            # LLM at least has the sources.
            listing = "\n".join(f"- {u}" for u in links[:top_k])
            self.last_sources = list(links[:top_k])
            return f"Web search results for: {query}\n\n{listing}"

        ranked = self._rerank(query, fetched)
        kept = ranked[:top_k]
        self.last_sources = [url for url, _ in kept]
        parts = [f"Web search results for: {query}"]
        for i, (url, content) in enumerate(kept, 1):
            parts.append(f"## [{i}] {url}\n\n{content}")
        logger.info(
            "[web_search] fetched %d/%d usable source(s), kept top %d (%d skipped) for %r",
            len(fetched), num_candidates, len(kept), len(skipped), query,
        )
        return "\n\n".join(parts)

    def _rerank(self, query: str, candidates: List[tuple]) -> List[tuple]:
        """Re-order *candidates* (url, content) by LLM-judged relevance.

        Asks the model for a comma-separated ranking of the candidate indices
        and re-orders accordingly.  Falls back to the input (endpoint) order
        when no LLM is wired in or the call fails, so web search never degrades
        below the positional ranking.
        """
        if not candidates or self.llm is None or len(candidates) == 1:
            return candidates
        # Judge relevance on a short snippet of each source — full content is
        # too much for the rerank model's window and adds no signal.
        lines = []
        for i, (url, content) in enumerate(candidates, 1):
            snippet = re.sub(r"\s+", " ", content)[:400]
            lines.append(f"{i}. {url} — {snippet}")
        prompt = (
            "Rank the web results below by relevance to the user's query, most "
            'relevant first. Respond with ONLY the numbers as a comma-separated '
            'list, e.g. "3,1,5". No explanation.\n\n'
            f"Query: {query}\n\n" + "\n".join(lines)
        )
        try:
            resp = str(self.llm.invoke(prompt, max_tokens=32)).strip()
            order = [int(x) for x in re.findall(r"\d+", resp)]
        except Exception as e:
            logger.info("[web_search] LLM rerank failed (%s) — using endpoint order", e)
            return candidates
        seen: set = set()
        ranked: List[tuple] = []
        for i in order:
            if 1 <= i <= len(candidates):
                url = candidates[i - 1][0]
                if url not in seen:
                    seen.add(url)
                    ranked.append(candidates[i - 1])
        for c in candidates:  # any candidates the model skipped stay at the end
            if c[0] not in seen:
                seen.add(c[0])
                ranked.append(c)
        logger.info("[web_search] LLM reranked %d source(s) for %r", len(ranked), query)
        return ranked

    @staticmethod
    def _is_usable(content: str) -> bool:
        """True when fetched content is real page text (not empty / JS-only)."""
        if not content or not content.strip():
            return False
        low = content.lower()
        if "requires javascript" in low or "javascript-rendered pages cannot" in low:
            return False
        return len(content.strip()) >= 200

    # ------------------------------------------------------------------ config

    @staticmethod
    def is_enabled() -> bool:
        """True when web search is enabled on the current project."""
        try:
            from smartloop.workspace.workspace import Workspace

            ws = Workspace._instance or Workspace.load()
            project = ws.current_project if ws else None
            return bool(project and project.web_search_enabled)
        except Exception as e:
            logger.warning("[web_search] enablement check failed: %s", e)
            return False
