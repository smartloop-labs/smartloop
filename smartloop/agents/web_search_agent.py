"""Web search agent — pulls the latest information and events from the web.

This is a placeholder implementation.  The concrete search backend/API is
still to be defined; until then the agent simply advertises that it can
process up-to-date information so the planner can route "latest news / recent
events" style prompts to it.  It runs as an ordinary plan step
(``StepType.WEB_SEARCH``) resolved by the orchestrator, mirroring the
``FetchAgent`` contract (``search()`` returns markdown-ish text for the step).
"""

import logging

logger = logging.getLogger(__name__)

_PLACEHOLDER_RESULT = (
    "Web search is available and can process the latest information and "
    "current events. (The web search backend is not yet configured, so no "
    "live results were retrieved for this query.)"
)


class WebSearchAgent:
    """Retrieve up-to-date information from the web for a query.

    The real backend (search API + result fetching) is TBD; see the module
    docstring.  ``search`` currently returns a fixed capability notice so the
    rest of the pipeline (planning, reasoning) can integrate against a stable
    interface now and gain live results once the backend lands.
    """

    def search(self, query: str, max_results: int = 5) -> str:
        """Return web results for *query* as text.

        Args:
            query: Natural-language search query.
            max_results: Maximum number of results to incorporate (reserved
                for the real backend).
        """
        logger.info("[web_search] query=%r (placeholder backend)", query)
        return _PLACEHOLDER_RESULT
