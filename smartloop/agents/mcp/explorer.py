"""
Explorer — unified system tool for exploring skills, vector search,
conversation history/context, filesystem walk, and URL fetch.
"""
import logging
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartloop.workspace.context import ProjectBoundExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from smartloop.llms import BaseLLM

from smartloop.agents.planner import PlanStep, StepType, MCPToolContext, SYSTEM_TOOLS_SERVER_ID

from .server_selector import (
    EXPLORE_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL, BASH_TOOL,
)
from .mcp_tool_caller import MCPToolCaller
from .argument_extractor import MCPArgumentExtractor
from .mcp_agent import (
    MCPAgent, format_call, is_entitlement_notice, mark_tool_unavailable,
    missing_required,
)

if TYPE_CHECKING:
    from smartloop.workspace.workspace import MCPServer

logger = logging.getLogger(__name__)

# Code-defect exceptions re-raise (surface bugs) rather than degrading
# a search to empty; genuine runtime/IO errors are still swallowed.
_PROGRAMMING_ERRORS = (NameError, AttributeError, TypeError, ImportError)

# ------------------------------------------------------------------ constants

FETCH_TOOL_KEYWORDS = {"fetch", "browse", "scrape", "web", "http", "navigate", "get_url"}

EXPLORE_EXTENSIONS = {
    ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini",
    ".md", ".txt", ".rst",
    ".html", ".css", ".sql", ".proto", ".graphql",
    ".sh",
}
EXPLORE_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info", ".out",
}
MAX_EXPLORE_FILES = 50
MAX_FILE_CHARS = 2000

# How much of one fetched record body reaches the prompt.  A Notion page can be
# very long and three of them share the answer's context with everything else
# collect gathered.
_MAX_RECORD_BODY_CHARS = 4000

MAX_SKILL_FILE_CHARS = 4_000
MAX_TREE_ENTRIES = 10
MAX_HISTORY_TURNS = 8
MAX_PROJECT_FILES = 30
MAX_PROJECT_SNIPPET_LINES = 30
MAX_PROJECT_FILES_CHARS = 8_000
MAX_FILE_SNIPPET_CHARS = 3_000

_SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", "venv", ".venv",
    "dist", "build", ".tox", ".mypy_cache", ".ruff_cache", ".out",
})

_PROJECT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".yaml", ".yml", ".toml",
    ".json", ".md", ".txt", ".cfg", ".ini",
    ".html", ".css", ".sql", ".proto", ".graphql",
    ".sh",
}

_QUERY_STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from",
    "what", "how", "does", "project", "explore", "analyze",
    "explain", "summarize", "understand", "codebase", "code",
}


# ------------------------------------------------------------------ models

class SearchContext(BaseModel):
    """Collected explore results: vector, feedback, workspace, and web search."""

    vector_context: str = ""
    feedback_context: str = ""
    workspace_context: str = ""
    # Latest-info content from the hosted web search (Pro feature). Gathered as
    # part of explore — combined with local folder / workspace / vector results.
    web_context: str = ""
    # URLs of the web sources actually used (top reranked links), so the
    # orchestrator can surface them as links in the final answer.
    web_sources: List[str] = Field(default_factory=list)
    # url -> the fetched page text that source contributed to the prompt, so a
    # web reference is only credited when the answer actually drew on it.
    web_source_extracts: Dict[str, str] = Field(default_factory=dict)
    # Result of an auto-run read-only MCP tool (a "collect" step). Write /
    # side-effecting tools are NOT run here — they stay planned execute steps.
    mcp_context: str = ""
    citations: Optional[List[Dict[str, Any]]] = None
    # Per-document search results ({"document_name", "content"}), ordered by
    # relevance.  Populated by the parallel vector search and consumed by the
    # orchestrator's document fan-out (one DAG node per document/batch).
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    # Document names the fan-out kept (judged relevant).  When set, the final
    # citations are restricted to these so an entity query cites only the
    # documents that actually contributed to the answer.  None = no fan-out
    # filtering applied (cite everything retrieved).
    cited_documents: Optional[List[str]] = None

    @property
    def has_context(self) -> bool:
        return bool(
            self.vector_context
            or self.feedback_context
            or self.workspace_context
            or self.web_context
            or self.mcp_context
        )


class EnrichedContext(BaseModel):
    """Aggregated exploration output consumed by the orchestrator."""

    skill_contents: List[str] = Field(default_factory=list)
    workspace_tree: str = Field(default="")
    mcp_catalogue: str = Field(default="")
    history_insight: str = Field(default="")
    project_files: str = Field(default="")
    analysis: str = Field(default="")
    exploration_steps: List[str] = Field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(
            self.skill_contents
            or self.workspace_tree
            or self.mcp_catalogue
            or self.history_insight
            or self.project_files
            or self.analysis
        )


# ------------------------------------------------------------------ explorer

class Explorer:
    """Unified explorer: context enrichment + filesystem/vector/URL exploration."""

    def __init__(
        self,
        llm: BaseLLM,
        tool_caller: Optional[MCPToolCaller] = None,
        arg_extractor: Optional[MCPArgumentExtractor] = None,
        *,
        use_llm_for_history: bool = False,
    ):
        self.llm = llm
        self.tool_caller = tool_caller
        self.arg_extractor = arg_extractor
        self._use_llm_history = use_llm_for_history
        from smartloop.agents.fetch_agent import FetchAgent
        self.fetch_agent = FetchAgent()
        # Web search (Pro) is part of explore: it reuses the fetch agent to pull
        # each result link's content, combined with vector/workspace/local.
        from smartloop.agents.web_search_agent import WebSearchAgent
        self.web_search_agent = WebSearchAgent(fetch_agent=self.fetch_agent, llm=llm)
        # MCP tool retrieval (vector) for the collect phase. Only read-only
        # tools are auto-run here; write tools stay planned execute steps.
        self.mcp_agent = MCPAgent(llm, logger)

    # ================================================================== #
    # Skill matching                                                       #
    # ================================================================== #

    def match_skills(
        self,
        user_message: str,
        parsed_skills: List[Dict[str, str]],
        llm: Optional[BaseLLM] = None,
    ) -> List[Dict[str, str]]:
        """Match a user message to skills, unioning two complementary signals.

        Keyword overlap alone is brittle — a real request ("what is the
        purpose of the funds?") may share only one word with the skill, and a
        complex prompt may share none.  So the LLM is the primary qualifier
        and handles indirect / multi-part phrasing; a cheap deterministic
        *direct* topic match against the curated skill name backs it up so an
        obvious hit ("funds" ↔ ``fund-info``) is never lost to a flaky small
        model.

        ``llm`` overrides the model used for the semantic qualifier — the
        caller passes the ``skills``-capability model, which is far better at
        the intent judgement than the small orchestrator model.  Falls back to
        ``self.llm`` when not given.
        """
        matchable = [
            s for s in parsed_skills
            if not s["name"].startswith("__")
        ]
        if not matchable:
            return []

        direct = self._direct_name_matches(user_message, matchable)
        llm_matched = self._llm_match_skills(user_message, matchable, llm=llm)

        # Union, preserving the presentation order; dedup by skill name.
        keep = {s["name"] for s in direct} | {s["name"] for s in llm_matched}
        matched = [s for s in matchable if s["name"] in keep]

        if matched:
            logger.info(
                f"[skill-match] Matched {len(matched)} skill(s): "
                f"{[s['name'] for s in matched]} "
                f"(direct={[s['name'] for s in direct]})"
            )
        return matched

    def _llm_match_skills(
        self,
        user_message: str,
        matchable: List[Dict[str, str]],
        llm: Optional[BaseLLM] = None,
    ) -> List[Dict[str, str]]:
        """Ask the LLM which skills apply — the primary, semantic matcher.

        Handles complex or indirectly-worded prompts that share few or no
        literal keywords with the skill.  Returns ``[]`` when the model
        matches nothing or the call fails (the deterministic direct-name
        match in :meth:`match_skills` still stands on its own).
        """
        skill_list = "\n".join(
            f"  {i+1}. {s['name']}: {s['description']}"
            for i, s in enumerate(matchable)
        )

        prompt = (
            "<system>You decide which instruction sets (skills), if any, apply "
            "to the user's request. Judge by INTENT, not topic: match a skill "
            "only when the request asks you to perform the task the skill is "
            "for. A general or definitional question ('what is X', 'why does X "
            "matter', 'what is the purpose of X') is NOT a match for a skill "
            "that acts on X (e.g. summarizes or reports X).\n\n"
            "Answer with the matching skill number(s), comma-separated. If none "
            "apply, answer exactly -1. Prefer -1 when unsure; never pick a "
            "skill just because it is the only one listed.\n\n"
            "Examples — given skill '1. sales-report: build a summary table of "
            "sales figures':\n"
            "  request 'summarize this quarter's sales' -> 1\n"
            "  request 'what is a sales report?' -> -1\n"
            "  request 'what is the purpose of sales?' -> -1\n\n"
            f"Skills:\n{skill_list}</system>"
            f"<user>{user_message}</user>"
        )

        match_llm = llm or self.llm
        try:
            # Skill matching only needs the model to emit a few skill
            # numbers (e.g. "1, 3"). Cap the output so a missing stop token
            # can't trigger runaway greedy generation that stalls the request.
            response = str(match_llm.invoke(prompt, max_tokens=16)).strip().lower()
            logger.info(f"[skill-match] LLM response: {response[:100]}")
        except Exception as e:
            logger.warning(f"[skill-match] LLM failed: {e}")
            return []

        # Explicit no-match sentinels. Handle these before digit parsing so
        # "-1" isn't misread as skill 1 (``\d+`` would match the "1").
        if "none" in response and len(response) < 10:
            return []

        matched: List[Dict[str, str]] = []
        seen: set = set()

        # Parse signed integers (handles "1", "1, 3", "-1"). Only positive
        # 1-based indices select a skill; -1 / 0 / out-of-range are ignored,
        # so a "-1" reply yields no matches.
        for token in re.findall(r'-?\d+', response):
            idx = int(token) - 1  # 1-based in prompt
            if 0 <= idx < len(matchable):
                s = matchable[idx]
                if s["name"] not in seen:
                    matched.append(s)
                    seen.add(s["name"])

        # Fallback: the model sometimes echoes skill names instead of numbers.
        if not matched:
            for s in matchable:
                if s["name"].lower() in response and s["name"] not in seen:
                    matched.append(s)
                    seen.add(s["name"])

        return matched

    @staticmethod
    def _direct_name_matches(
        user_message: str,
        skills: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Match skills the user *names explicitly* — a direct invocation.

        This is intentionally strict: it fires only when the skill's full
        name appears as a phrase in the message (e.g. "run the fund-info
        skill", "give me the fund info").  A bare topic mention ("what is
        the purpose of the funds?") is NOT a direct match — that judgement
        belongs to the LLM qualifier, which weighs the user's intent against
        the skill's described use case.
        """
        msg = re.sub(r'[\s_\-]+', ' ', user_message.lower()).strip()

        matched: List[Dict[str, str]] = []
        for s in skills:
            name = s.get("name", "")
            phrase = re.sub(r'[\s_\-]+', ' ', name.lower()).strip()
            if not phrase:
                continue
            # Word-boundary phrase match so "fund info" hits but "funding fo"
            # does not.
            if re.search(rf'\b{re.escape(phrase)}\b', msg):
                logger.debug(f"[skill-match] direct name invocation: '{name}'")
                matched.append(s)

        return matched

    # ================================================================== #
    # Parallel search primitives                                          #
    # ================================================================== #

    @staticmethod
    def _search_vector(
        query: str,
        model_service: Any,
        skill_instruction: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> Optional[tuple]:
        """Vector/document search — thread-safe, I/O bound.

        Returns ``(text, citations, documents)`` where ``documents`` is the
        per-document list used for fan-out, or ``None`` when no results.
        """
        try:
            result = model_service.get_document_groups(
                query=query, skill_instruction=skill_instruction,
                conversation_history=conversation_history,
            )
            if result is not None:
                documents, citations = result
                text = "\n\n---\n\n".join(
                    f"=== Document: {d['document_name']} ===\n{d['content']}"
                    for d in documents
                )
                logger.info(
                    f"[collect] Vector search returned {len(text)} chars"
                    f" from {len(documents)} document(s)"
                )
                return text, citations, documents
            logger.info("[collect] Vector search returned no results")
        except Exception as e:
            if isinstance(e, _PROGRAMMING_ERRORS):
                raise
            logger.warning(f"[collect] Vector search failed: {e}")
        return None

    @staticmethod
    def _search_feedback(query: str, model_service: Any) -> str:
        """Feedback Q&A search — thread-safe, I/O bound."""
        try:
            feedback_chunks = model_service.search_feedback(query, n_results=5)
            if feedback_chunks:
                text = "\n---\n".join(c.text for c in feedback_chunks)
                logger.info(
                    f"[collect] Feedback search returned {len(feedback_chunks)} Q&A pair(s)"
                )
                return text
        except Exception as e:
            if isinstance(e, _PROGRAMMING_ERRORS):
                raise
            logger.warning(f"[collect] Feedback search failed: {e}")
        return ""

    # How much of the transcript the follow-up gate sees.  The decision only
    # needs the shape of the exchange — what was asked, what came back — not
    # the full text of a long answer.
    _GATE_TURNS = 4
    _GATE_TURN_CHARS = 400
    _GATE_MAX_TOKENS = 8

    def _needs_web_search(
        self, query: str, conversation_history: Optional[List[Dict[str, str]]],
    ) -> bool:
        """True when *query* needs fresh web results rather than the transcript.

        A follow-up that only reworks what the assistant already produced
        ("now create a table", "summarise that", "translate it") is answerable
        from the conversation.  Searching again for it is pure noise: the query
        carries none of the original subject, so the results come back about
        tables rather than about whatever was being tabulated, and those pages
        land in the context and as citations.

        The first message of a session always searches — there is nothing to
        rework yet.  Fails open: an LLM error or an unreadable verdict
        searches, so this can only suppress a search it is confident about.
        """
        if not conversation_history or self.llm is None:
            return True

        turns = [t for t in conversation_history if t.get("content")]
        if not turns:
            return True
        transcript = "\n".join(
            f"{t.get('role', 'user')}: {t['content'][:self._GATE_TURN_CHARS]}"
            for t in turns[-self._GATE_TURNS:]
        )

        prompt = f"""\
<system>
You decide whether a new web search is needed to answer the user's latest message.

Answer SEARCH when the message asks for information that is not already in the
conversation: a new topic, different subject, more recent data, or a detail the
previous answer did not cover.

Answer REUSE when the message only asks to rework what the assistant already
produced — reformat it (table, list, bullets), shorten it, expand it, sort it,
translate it, or explain it. Reworking needs no new information.

Respond with exactly one word: SEARCH or REUSE.
</system>
<user>
Conversation so far:
{transcript}

Latest message: {query}

Does answering the latest message require a new web search?
</user>"""

        try:
            verdict = str(
                self.llm.invoke(prompt, max_tokens=self._GATE_MAX_TOKENS)
            ).strip().upper()
        except Exception as exc:
            logger.warning("[collect] Web-search gate failed, searching: %s", exc)
            return True

        if "REUSE" in verdict and "SEARCH" not in verdict:
            logger.info(
                "[collect] Skipping web search — %r reworks the previous "
                "answer, no new information needed", query[:80],
            )
            return False
        return True

    def _search_web(
        self,
        query: str,
        progress: Optional[Callable[[str, str, str], None]] = None,
    ) -> str:
        """Hosted web search (Pro), part of explore — thread-safe, I/O bound.

        Runs the WebSearchAgent, which fetches each result link's content. Any
        failure is a silent skip (returns ``""``) so the model falls back to its
        own knowledge — no error/upsell is injected.

        ``progress`` is forwarded so each page is reported as it is fetched.
        """
        try:
            content = self.web_search_agent.search(query, progress=progress)
            if content:
                logger.info(f"[collect] Web search returned {len(content)} chars")
            return content or ""
        except Exception as e:
            logger.warning(f"[collect] Web search failed: {e}")
            return ""

    # Verbs that mark a tool as read-only (safe to auto-run during collect).
    _READ_ONLY_VERBS = {
        "get", "list", "search", "read", "fetch", "find", "describe", "view",
        "show", "query", "lookup", "retrieve", "count", "check", "status",
        "browse", "scrape",
    }
    # Verbs that mark a tool as side-effecting (never auto-run in collect).
    _WRITE_VERBS = {
        "create", "update", "delete", "send", "post", "write", "remove", "set",
        "add", "edit", "modify", "put", "patch", "insert", "cancel", "pay",
        "deploy", "merge", "close", "assign", "run", "execute", "trigger",
    }

    @classmethod
    def _is_read_only_tool(cls, tool_name: str) -> bool:
        """True only when a tool name clearly reads (no side effects).

        Every word in the name is considered, not just the first.  Servers
        name tools after the product as often as after the verb —
        ``notion-search``, ``API-get-users``, ``trelloReadCard`` — and a
        leading-verb rule read "notion" as an unknown verb and refused to run
        the one tool the question was about.  Any write word anywhere still
        vetoes: ``notion-create-pages`` is a write however it is spelled.

        Conservative in both directions: a name with no recognised verb is not
        read-only, so a side-effecting tool is never auto-run during collect —
        it stays a planned execute step.  (MCP annotations aren't persisted, so
        this is a heuristic over the name.)
        """
        if not tool_name:
            return False
        # camelCase is a word boundary too: `trelloReadCard` is three words.
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", tool_name.strip())
        words = set(re.split(r"[_\-\s.]+", spaced.lower()))
        if words & cls._WRITE_VERBS:
            return False
        return bool(words & cls._READ_ONLY_VERBS)

    # How many connected servers are asked for one question.  Each costs a
    # reasoned resolution and a network round-trip, and a workspace with a
    # dozen connections would put all of them in front of every answer.
    _MAX_SERVERS_PER_QUESTION = 4

    @staticmethod
    def _emit(
        progress: Optional[Callable[[str, str, str], None]],
        step: str,
        message: str,
    ) -> None:
        """Report one step to the client, never failing the work over it.

        A progress callback crosses a thread boundary into the request's event
        loop (see ``chat.py``), so it is the kind of thing that can raise long
        after the client has gone.  Collect's job is the context, not the
        commentary.
        """
        if progress is None:
            return
        try:
            progress(step, "processing", message)
        except Exception:
            logger.debug("[collect] progress callback failed", exc_info=True)

    def _run_mcp(self, query: str, mcp_servers: Optional[List["MCPServer"]],
                 project_dir: Optional[str], llm: Optional[BaseLLM] = None,
                 progress: Optional[Callable[[str, str, str], None]] = None) -> str:
        """Gather what the connected servers know about *query*.

        Every server is asked, not just the best-matching one: a question is
        put to documents, workspace files, feedback and the web all at once,
        and a connected server is a source like any of them.  Resolving over
        the whole catalog instead picked one tool for the whole workspace, so a
        question spanning two servers came back with half its context — and on
        a large catalog the pick was made from a list too long to describe.

        Per server: resolve a call over *its* tools, run it when the tool is
        read-only, then follow the connection's call plan to fetch the bodies
        behind the records it returned (see :mod:`smartloop.mcp.call_plans`).
        Write/side-effecting tools are left for a planned execute step.  A
        failure anywhere is a silent skip for that server, never for the rest.
        """
        if not self.tool_caller or not self.arg_extractor or not mcp_servers:
            return ""

        servers = [
            s for s in mcp_servers
            if getattr(s, "enabled", True)
            and not getattr(s, "system", False)
            and getattr(s, "id", None) != SYSTEM_TOOLS_SERVER_ID
            and (getattr(s, "tools", None) or [])
        ]
        if not servers:
            return ""
        if len(servers) > self._MAX_SERVERS_PER_QUESTION:
            logger.info(
                "[collect] %d connected servers — asking the first %d",
                len(servers), self._MAX_SERVERS_PER_QUESTION,
            )
            servers = servers[: self._MAX_SERVERS_PER_QUESTION]

        gathered = []
        for server in servers:
            text = self._run_mcp_server(query, server, llm, progress=progress)
            if text:
                gathered.append(text)
        return "\n\n".join(gathered)

    def _run_mcp_server(
        self, query: str, server: "MCPServer", llm: Optional[BaseLLM] = None,
        progress: Optional[Callable[[str, str, str], None]] = None,
    ) -> str:
        """One server's answer to *query*, or ``""``."""
        server_name = getattr(server, "name", None) or "the server"
        try:
            # Resolution is the long leg — a reasoned pick over this server's
            # whole tool list, tens of seconds on a large catalog — and it runs
            # before anything is called, so with no event here the client sits
            # on "Processing..." through all of it and the work reads as a
            # hang.  Named per server, because a workspace with two
            # connections pays this once each.
            # Two attempts, because the first pick can turn out to be a tool
            # this account cannot call.  Discovery lists what the server has,
            # not what the plan is entitled to, so a paywalled tool is
            # indistinguishable from a usable one until it answers — and it
            # answers with an advertisement, successfully.  Retrying once with
            # it struck off the catalogue is what lets the same turn fall to
            # the tool that does work (``notion-search`` where
            # ``notion-ai-search`` is Business-only) instead of handing the
            # upsell to the model as though it were the document.
            raw = ""
            tool_name = ""
            for attempt in (1, 2):
                self._emit(progress, "tool", f"Choosing a {server_name} tool")
                picked = self.mcp_agent.resolve(query, [server], llm=llm)
                if not picked:
                    return ""
                tool_name = picked.tool_name
                if not self._is_read_only_tool(tool_name):
                    logger.info(
                        "[collect] MCP tool '%s' is not read-only — leaving to a "
                        "planned execute step", tool_name,
                    )
                    return ""

                # The tool and its arguments, not just the server: "Calling
                # Notion" is indistinguishable between reading a page and
                # searching for one, and which of the two ran is the thing a
                # user watching this actually wants to know.
                self._emit(
                    progress, "tool",
                    f"Calling {format_call(server_name, tool_name, picked.arguments)}",
                )
                raw = self._call_mcp_tool(server, tool_name, picked.arguments, query)
                if not raw:
                    return ""

                if not is_entitlement_notice(raw):
                    break

                # Struck off before the retry, so the same resolution cannot
                # come back with the same answer.
                mark_tool_unavailable(getattr(server, "id", "") or "", tool_name)
                logger.warning(
                    "[collect] %s.%s is not available on this account — %s. "
                    "Dropping it from the catalogue%s",
                    server_name, tool_name, raw.strip().splitlines()[0][:160],
                    " and re-resolving" if attempt == 1 else "",
                )
                self._emit(
                    progress, "tool",
                    f"{server_name}.{tool_name} needs a higher plan — "
                    "trying another tool",
                )
                raw = ""
            if not raw:
                return ""

            # A listing answers "which records", not "what do they say".  The
            # connection's plan names the tool that fetches a record's body, so
            # a search hit becomes the page behind it rather than its title.
            detail = self._fetch_record_details(
                server, tool_name, raw, progress=progress,
            )
            body = f"{raw}\n\n{detail}" if detail else raw

            logger.info(
                "[collect] %s.%s returned %d chars",
                getattr(server, "name", ""), tool_name, len(body),
            )
            # The payload, not just its size.  A short answer here is the
            # difference between "the tool found nothing", "the tool answered
            # in prose that carries no record ids to follow up on" and "the
            # follow-up plan missed" — three different bugs that a char count
            # cannot tell apart, and the execute path has always logged its
            # raw result for exactly this reason.
            logger.info(
                "[collect] %s.%s raw result: %s",
                getattr(server, "name", ""), tool_name, body[:1000],
            )
            return f"MCP TOOL ({getattr(server, 'name', '')}.{tool_name}):\n{body}"
        except Exception as e:
            if isinstance(e, _PROGRAMMING_ERRORS):
                raise
            logger.warning(
                "[collect] MCP call on '%s' failed, skipping: %s",
                getattr(server, "name", ""), e,
            )
            return ""

    def _call_mcp_tool(
        self,
        server: "MCPServer",
        tool_name: str,
        arguments: Dict[str, Any],
        query: str,
    ) -> str:
        """Run one read-only tool and return its text, or ``""``."""
        tool_def = next(
            (t for t in (server.tools or []) if getattr(t, "name", None) == tool_name),
            None,
        )
        input_schema = (getattr(tool_def, "input_schema", None) or {}) if tool_def else {}

        # Resolution filled the required arguments in the same call that picked
        # the tool; the extractor is only needed when it could not.
        arguments = dict(arguments or {})
        if missing_required(input_schema, arguments):
            arguments = self.arg_extractor.invoke(
                user_input=query, tool_name=tool_name, input_schema=input_schema,
            )
            arguments.pop("_missing_params", None)

        # The call itself, in the log, before it goes out: a resolution that
        # picked a tool and a call that actually reached the server are
        # different events, and only the second one can fail.
        logger.info(
            "[collect] calling %s",
            format_call(getattr(server, "name", ""), tool_name, arguments),
        )
        result = self.tool_caller.call_tool(
            server=server, tool_name=tool_name, arguments=arguments,
            access_token=self.tool_caller.get_access_token(server),
        )
        if not result.success or not result.content:
            logger.info(
                "[collect] %s returned nothing (%s)",
                tool_name, getattr(result, "error", None) or "empty result",
            )
            return ""
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) and block.get("type") == "text"
            else str(block)
            for block in result.content
        ).strip()

    def _fetch_record_details(
        self, server: "MCPServer", tool_name: str, payload: str,
        progress: Optional[Callable[[str, str, str], None]] = None,
    ) -> str:
        """Bodies behind the records *payload* lists, per the connection's plan.

        Returns ``""`` when the connection has no plan, the plan names no
        detail tool the server has, or the payload carries no record ids —
        every one of which is the ordinary case for a server whose listing is
        already the answer.
        """
        try:
            from smartloop.mcp.call_plans import follow_up_calls

            calls = follow_up_calls(server, tool_name, payload)
        except Exception as e:
            logger.debug("[collect] no follow-up plan for %s: %s", tool_name, e)
            return ""
        if not calls:
            return ""

        server_name = getattr(server, "name", None) or "the server"
        bodies = []
        # Announced per record, not once for the batch: a search hit list turns
        # into one fetch each, and a question that matched five pages spends
        # most of its collect here.  Reported with the same call format as the
        # resolved tool, so the follow-ups read as the calls they are rather
        # than as a single opaque step.
        for index, (detail_tool, arguments) in enumerate(calls, start=1):
            logger.info(
                "[collect] following up with %s",
                format_call(getattr(server, "name", ""), detail_tool, arguments),
            )
            self._emit(
                progress, "tool",
                f"Reading {index} of {len(calls)} from {server_name}: "
                f"{format_call('', detail_tool, arguments)}",
            )
            result = self.tool_caller.call_tool(
                server=server, tool_name=detail_tool, arguments=arguments,
                access_token=self.tool_caller.get_access_token(server),
            )
            # The other three call paths renew an expired token and try again;
            # this one returned "" instead, so a connected app whose token had
            # simply aged out looked to retrieval like an app with nothing in it.
            if not result.success and result.auth_expired:
                new_token = self.tool_caller.refresh_access_token(server)
                if new_token:
                    result = self.tool_caller.call_tool(
                        server=server, tool_name=tool_name,
                        arguments=arguments, access_token=new_token,
                    )
            if not result.success or not result.content:
                continue
            text = "\n".join(
                block.get("text", "") if isinstance(block, dict) and block.get("type") == "text"
                else str(block)
                for block in result.content
            ).strip()
            if text:
                bodies.append(text[:_MAX_RECORD_BODY_CHARS])
        if not bodies:
            return ""
        logger.info(
            "[collect] fetched %d record body/bodies from %s",
            len(bodies), getattr(server, "name", ""),
        )
        return "\n\n".join(bodies)

    @staticmethod
    def _is_non_data_file(fname: str) -> bool:
        """True for skill/system files that are instructions or tool metadata,
        not user data — they must never leak into retrieved context.

        Covers skill files (``skills/`` dirs, ``SKILL.md``), MCP catalogue files
        (``mcp-*``), and bundled system files whose basename is ``__``-prefixed —
        notably ``__system_mcp__.md``, which lists the built-in tool
        descriptions. Without this a fund/table prompt can sweep those tool
        descriptions in as fake data rows (e.g. "Smartloop | SMTL | N/A").
        """
        if not fname:
            return False
        base = os.path.basename(fname)
        return (
            "/skills/" in fname
            or fname.startswith("skills/")
            or fname.upper().endswith("SKILL.MD")
            or fname.startswith("mcp-")
            or base.startswith("mcp-")
            or base.startswith("__")
        )

    @staticmethod
    def _search_workspace(
        query: str,
        model_service: Any,
        cwd: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Workspace file search — thread-safe, I/O bound."""
        try:
            ws_chunks = model_service.search_workspace(
                query, cwd, n_results=10,
                conversation_history=conversation_history,
            )
            if ws_chunks:
                seen_files: dict = {}
                for chunk in ws_chunks:
                    fname = chunk.document_name or chunk.document_id
                    if fname and Explorer._is_non_data_file(fname):
                        continue
                    if fname not in seen_files or chunk.distance < seen_files[fname].distance:
                        seen_files[fname] = chunk
                ws_texts = []
                for fname, chunk in seen_files.items():
                    ws_texts.append(f"--- {fname} ---\n{chunk.text[:2000]}")
                text = "\n".join(ws_texts)
                logger.info(
                    f"[collect] Workspace search returned {len(seen_files)} file(s)"
                )
                return text
        except Exception as e:
            if isinstance(e, _PROGRAMMING_ERRORS):
                raise
            logger.warning(f"[collect] Workspace search failed: {e}")
        return ""

    def collect_parallel(
        self,
        query: str,
        model_service: Optional[Any] = None,
        cwd: Optional[str] = None,
        enriched: Optional[EnrichedContext] = None,
        matched_skills: Optional[List[Dict]] = None,
        mcp_servers: Optional[List["MCPServer"]] = None,
        mcp_tool_contexts: Optional[List["MCPToolContext"]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        project_dir: Optional[str] = None,
        progress: Optional[Callable[[str, str, str], None]] = None,
        mcp_llm: Optional[BaseLLM] = None,
    ) -> SearchContext:
        """Fan-out vector/feedback/workspace searches in parallel.

        Drop-in replacement for :meth:`collect` — same signature (plus
        optional ``executor``).  The three I/O-bound searches run
        concurrently via a :class:`ThreadPoolExecutor`.  Skill tool
        execution still runs sequentially afterward (tools may depend
        on search results).

        ``progress(step, status, message)`` — when supplied — reports the web
        search leg.  The local searches are fast and finish together; the web
        search is the one the user waits on.
        """
        ctx = SearchContext()

        if not model_service:
            return ctx

        # Project-bound: the fan-out searches below resolve against the
        # requesting project (see smartloop.workspace.context).
        pool = executor or ProjectBoundExecutor(max_workers=6)
        own_pool = executor is None

        try:
            # When a skill matched, the pipeline is self-contained — do not
            # bias the vector search with previous conversation turns.  The
            # current user message is the only input that matters.
            history_for_search = None if matched_skills else conversation_history

            f_vec = pool.submit(
                self._search_vector, query, model_service,
                None, history_for_search,
            )
            f_fb = pool.submit(self._search_feedback, query, model_service)
            f_ws = pool.submit(
                self._search_workspace, query, model_service, cwd,
                history_for_search,
            ) if cwd else None
            # MCP (read-only): part of explore, and on the same footing as
            # every other source — the connected servers are searched for this
            # question alongside documents, workspace files, feedback and the
            # web, in the same fan-out.  The tool is resolved by reasoning over
            # what discovery found (MCPAgent) and run only if it is read-only;
            # write tools stay planned execute steps.
            #
            # This used to be gated on a vectorized copy of the server's
            # content: when that local index matched, the live call was skipped
            # and the answer came from a snapshot. The snapshot is gone, so the
            # source of truth is the server, asked now.
            f_mcp = None
            if mcp_servers:
                f_mcp = pool.submit(
                    self._run_mcp, query, mcp_servers, project_dir, mcp_llm,
                    progress,
                )

            # Collect primary results (local retrieval + MCP tools)
            vec_result = f_vec.result()
            if vec_result is not None:
                ctx.vector_context = vec_result[0]
                ctx.citations = vec_result[1]
                ctx.documents = vec_result[2]

            ctx.feedback_context = f_fb.result()

            if f_ws is not None:
                ctx.workspace_context = f_ws.result()

            if f_mcp is not None:
                ctx.mcp_context = f_mcp.result()

            # Web search runs alongside local retrieval, not behind it.
            # Gating it on "did anything match locally?" made a single indexed
            # document or stored feedback Q&A — retrieval always returns its
            # nearest neighbours, related or not — silently cancel the search,
            # leaving the model to answer a camera question from cat notes.
            # Deciding which of the collected sources actually answers the
            # question is the synthesis step's job; collect just gathers.
            #
            # That applies to a live MCP tool result too. It used to cancel the
            # web search outright, on the reasoning that a deliberately invoked
            # tool had already answered — but "deliberately invoked" only means
            # the router picked it, not that its output answered the question,
            # and a query spanning both ("how does our roadmap compare to what
            # they announced") lost half its context. Both are gathered now and
            # the ordering above puts the local one first.
            #
            # The remaining exception is a follow-up that only reworks the
            # answer already on screen ("now make it a table") — see
            # :meth:`_needs_web_search`.
            if self.web_search_agent.is_enabled():
                if self._needs_web_search(query, conversation_history):
                    # Announce the search before it starts.  It is the longest
                    # leg of collect — a provider round-trip plus one fetch per
                    # result — and it runs on this thread while the local
                    # searches above have already returned, so with no event
                    # here the client shows the previous step until the whole
                    # search is done.  A skipped search stays silent, as the
                    # rest of this path is.
                    self._emit(
                        progress, "web_search",
                        "Searching the web for the latest information",
                    )
                    ctx.web_context = self._search_web(query, progress=progress)
                    if self.web_search_agent.last_sources:
                        ctx.web_sources = list(self.web_search_agent.last_sources)
                        ctx.web_source_extracts = dict(
                            getattr(self.web_search_agent, "last_source_extracts", None) or {}
                        )

        finally:
            if own_pool:
                pool.shutdown(wait=False)

        # Skill tool execution runs in the DAG step nodes — do NOT run them
        # here during collection.  Eager execution here would bypass argument
        # extraction, validation, and result accumulation in reason().

        return ctx

    # ================================================================== #
    # Context enrichment (pre-planning)                                    #
    # ================================================================== #

    def enrich(
        self,
        *,
        available_mcp_servers: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> EnrichedContext:
        ctx = EnrichedContext()

        if available_mcp_servers:
            ctx.mcp_catalogue = self._catalogue_mcp(available_mcp_servers)
            if ctx.mcp_catalogue:
                server_count = ctx.mcp_catalogue.count("\n") + 1
                ctx.exploration_steps.append(
                    f"MCP: catalogued {server_count} server(s)"
                )
                logger.info(
                    f"[explorer] MCP catalogue: {len(ctx.mcp_catalogue)} chars"
                )

        if conversation_history:
            ctx.history_insight = self._format_history(conversation_history)
            if ctx.history_insight:
                ctx.exploration_steps.append(
                    f"History: {len(conversation_history)} relevant turn(s)"
                )
                logger.info(
                    f"[explorer] history insight: {len(ctx.history_insight)} chars"
                )

        return ctx

    def analyze(
        self,
        user_message: str,
        enriched: EnrichedContext,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        skills: Optional[List[str]] = None,
    ) -> str:
        all_skills = list(skills or [])
        for sc in enriched.skill_contents:
            if sc not in all_skills:
                all_skills.append(sc)

        skill_summaries = []
        for sc in all_skills:
            name, desc = "", ""
            for line in sc.split("\n"):
                stripped = line.strip()
                if stripped.startswith("name:"):
                    name = stripped[5:].strip().strip("'\"")
                elif stripped.startswith("description:"):
                    desc = stripped[12:].strip().strip("'\"")
            if name:
                skill_summaries.append(f"  - {name}: {desc}" if desc else f"  - {name}")
        skills_block = "\n".join(skill_summaries) if skill_summaries else "None"

        mcp_block = enriched.mcp_catalogue or "None"
        history_block = enriched.history_insight or "None"
        workspace_block = enriched.workspace_tree or "None"
        project_block = enriched.project_files[:3000] if enriched.project_files else "None"

        prompt = (
            "<system>You are analyzing a user request. You have gathered "
            "context from indexed documents and skills.\n\n"
            "Skills are instructions that define how to handle specific types "
            "of requests. Each skill has a name and describes when it applies. "
            "Match the user request against ALL skills before deciding.\n\n"
            "Decide the route:\n"
            "- SKILL: A skill matches the user request. The skill's own "
            "instructions say how to respond (it may be conversational or "
            "may need a tool call).\n"
            "- DOCUMENT: The gathered document context answers the question. "
            "The skill may refine how to use the documents.\n"
            "- TOOL: An external MCP tool call is needed (fetch URL, API, etc.).\n"
            "- CONVERSATIONAL: General knowledge is sufficient.</system>\n"
            "<user>\n"
            f"User request: {user_message}\n\n"
            "---\n"
            f"SKILLS:\n{skills_block}\n\n"
            "---\n"
            f"DOCUMENT CONTEXT:\n{project_block}\n\n"
            "---\n"
            f"MCP TOOLS:\n{mcp_block}\n\n"
            "---\n"
            f"HISTORY:\n{history_block}\n"
        )

        prompt += (
            "\n---\n"
            "Answer briefly:\n"
            "1. What does the user want?\n"
            "2. Does a skill match? If yes, which skill name?\n"
            "3. Is the document context relevant to this request? (yes/no)\n"
            "4. Route: SKILL, DOCUMENT, TOOL, or CONVERSATIONAL\n"
            "</user>"
        )

        try:
            result = str(self.llm.invoke(prompt)).strip()
            logger.info(f"[analyze] LLM analysis ({len(result)} chars): {result[:300]}")
            return result
        except Exception as e:
            logger.warning(f"[analyze] Analysis failed (non-fatal): {e}")
            return ""

    # ================================================================== #
    # Enrichment internals (skills, workspace, history, project files)     #
    # ================================================================== #

    def _scan_skills(self, skills_dir: str) -> List[str]:
        contents: List[str] = []
        skills_path = Path(skills_dir)
        if not skills_path.is_dir():
            return contents

        candidates = list(skills_path.rglob("SKILL.md")) + list(
            p for p in skills_path.glob("*.md") if p.name != "SKILL.md"
        )
        for md_path in sorted(candidates):
            try:
                text = md_path.read_text(encoding="utf-8", errors="replace")
                trimmed = text[:MAX_SKILL_FILE_CHARS]
                if len(text) > MAX_SKILL_FILE_CHARS:
                    trimmed += "\n... (truncated)"
                if trimmed.strip():
                    contents.append(trimmed)
            except Exception as exc:
                logger.debug(f"[explorer] could not read skill {md_path}: {exc}")

        return contents

    def _scan_workspace(self, cwd: str) -> str:
        lines: List[str] = []
        cwd = os.path.realpath(cwd)

        try:
            for root, dirs, files in os.walk(cwd, topdown=True):
                dirs[:] = sorted(
                    d for d in dirs
                    if not d.startswith(".") and d not in _SKIP_DIRS
                )

                rel = os.path.relpath(root, cwd)
                level = 0 if rel == "." else rel.count(os.sep) + 1
                indent = "  " * level
                folder = os.path.basename(root) if rel != "." else cwd
                lines.append(f"{indent}{folder}/")

                sub_indent = "  " * (level + 1)
                for fname in sorted(files)[:20]:
                    lines.append(f"{sub_indent}{fname}")

                if len(lines) >= MAX_TREE_ENTRIES:
                    lines.append("  ... (truncated)")
                    break

        except PermissionError as exc:
            logger.debug(f"[explorer] workspace scan permission error: {exc}")
        except Exception as exc:
            logger.debug(f"[explorer] workspace scan failed: {exc}")

        return "\n".join(lines)

    def _catalogue_mcp(self, servers: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for srv in servers:
            srv_name = srv.get("name") or srv.get("id") or "unknown"
            tools = srv.get("tools") or []
            if not tools:
                continue
            names = [t.get("name", "?") for t in tools[:10]]
            if len(tools) > 10:
                names.append(f"… +{len(tools) - 10} more")
            lines.append(f"{srv_name}: {', '.join(names)}")
        return "\n".join(lines)

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        """Format recent history turns into a concise text block.

        The history is already selected by the caller (recency-based, via
        ``ModelService.get_relevant_history``), so this method simply
        renders each turn as ``role: content`` (truncated) without any
        keyword extraction or summarization.
        """
        if not history:
            return ""
        parts: List[str] = []
        for turn in history:
            role = turn.get("role", "unknown")
            content = (turn.get("content") or "").strip().replace("\n", " ")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _scan_project_files(self, cwd: str, user_message: str) -> str:
        cwd = os.path.realpath(cwd)

        candidates: List[str] = []
        try:
            for root, dirs, files in os.walk(cwd, topdown=True):
                dirs[:] = sorted(
                    d for d in dirs
                    if not d.startswith(".") and d not in _SKIP_DIRS
                    and not d.endswith(".egg-info")
                )
                for fname in sorted(files):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in _PROJECT_EXTENSIONS:
                        continue
                    rel = os.path.relpath(os.path.join(root, fname), cwd)
                    candidates.append(rel)
                    if len(candidates) >= MAX_PROJECT_FILES:
                        break
                if len(candidates) >= MAX_PROJECT_FILES:
                    break
        except Exception as exc:
            logger.debug(f"[explorer] project file walk failed: {exc}")
            return ""

        if not candidates:
            return ""

        query_terms = [
            w for w in re.findall(r"\w{3,}", user_message.lower())
            if w not in _QUERY_STOP_WORDS
        ]

        parts: List[str] = []
        total_chars = 0
        for fpath in candidates:
            full_path = os.path.join(cwd, fpath)
            try:
                content = open(full_path, "r", encoding="utf-8", errors="replace").read()
            except Exception:
                continue

            if not content.strip():
                continue

            snippet_lines: List[str] = []
            if query_terms:
                lines = content.split("\n")
                matched_indices: set = set()
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if any(term in line_lower for term in query_terms):
                        for j in range(max(0, i - 1), min(len(lines), i + 2)):
                            matched_indices.add(j)
                for i in sorted(matched_indices):
                    if len(snippet_lines) >= MAX_PROJECT_SNIPPET_LINES:
                        break
                    snippet_lines.append(f"{i+1}: {lines[i]}")

            if snippet_lines:
                snippet = "\n".join(snippet_lines)
            else:
                snippet = content[:MAX_FILE_SNIPPET_CHARS]
                if len(content) > MAX_FILE_SNIPPET_CHARS:
                    snippet += "\n... (truncated)"

            entry = f"\n--- {fpath} ---\n{snippet}"
            if total_chars + len(entry) > MAX_PROJECT_FILES_CHARS:
                break
            parts.append(entry)
            total_chars += len(entry)

        return "".join(parts)

    # ================================================================== #
    # Execution-time exploration (filesystem, vector, URL fetch)           #
    # ================================================================== #

    def explore(
        self,
        user_message: str,
        work_dir: str,
        project_dir: Optional[str] = None,
        mcp_servers: Optional[List["MCPServer"]] = None,
        model_service: Optional[Any] = None,
        vector_context: str = "",
        citations: Optional[list] = None,
        enriched: Optional[EnrichedContext] = None,
        skill_instruction: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> tuple[str, Optional[list]]:
        """Explore: URL fetch + collect (if not already done).

        When ``vector_context`` is already provided (caller ran collect()
        earlier), this method only adds URL fetching and assembles the
        raw context.  Otherwise it runs collect_parallel() internally.

        URL fetching and collect run concurrently when both are needed.

        Returns ``(raw_context, citations)`` — no LLM synthesis.
        The orchestrator's ``_format_ready_context()`` handles labelling
        and the final streaming LLM produces the user-facing response.
        """
        logger.info(f"[explore] Starting exploration in {work_dir}")

        urls = re.findall(r'https?://[^\s<>\'")\]]+', user_message)
        need_fetch = bool(urls and mcp_servers)
        need_collect = not vector_context

        # ── Parallel: fetch + collect when both needed ──
        fetch_context = ""
        feedback_context = ""
        summaries_text = ""

        if need_fetch and need_collect:
            pool = executor or ProjectBoundExecutor(max_workers=4)
            own_pool = executor is None
            try:
                f_fetch = pool.submit(self.fetch_url, urls[0], mcp_servers)
                f_collect = pool.submit(
                    self.collect_parallel, user_message,
                    model_service=model_service, cwd=work_dir,
                    conversation_history=conversation_history,
                    executor=pool,
                )
                fetch_context = f_fetch.result()
                search_ctx = f_collect.result()
                if search_ctx.vector_context:
                    vector_context = search_ctx.vector_context
                    citations = search_ctx.citations
                feedback_context = search_ctx.feedback_context
                summaries_text = search_ctx.workspace_context
                logger.info("[explore] Ran fetch + collect_parallel concurrently")
            finally:
                if own_pool:
                    pool.shutdown(wait=False)

            if not fetch_context and urls:
                fetch_context = (
                    f"Could not fetch content from {urls[0]}. The page may "
                    "require JavaScript or the fetch service is unavailable."
                )

        else:
            # ── Sequential paths (one or neither needed) ──
            if need_fetch:
                fetch_context = self.fetch_url(urls[0], mcp_servers)
                if not fetch_context and urls:
                    fetch_context = (
                        f"Could not fetch content from {urls[0]}. The page may "
                        "require JavaScript or the fetch service is unavailable."
                    )
                    logger.info("[explore] URL fetch returned empty")

            if need_collect:
                search_ctx = self.collect_parallel(
                    user_message, model_service=model_service,
                    cwd=work_dir,
                    conversation_history=conversation_history,
                    executor=executor,
                )
                if search_ctx.vector_context:
                    vector_context = search_ctx.vector_context
                    citations = search_ctx.citations
                feedback_context = search_ctx.feedback_context
                summaries_text = search_ctx.workspace_context
                logger.info("[explore] Ran collect_parallel() for vector/feedback/workspace")
            else:
                logger.info("[explore] Skipping collect() — vector_context already provided")

        # ── Assemble raw context ──
        context_parts: List[str] = []
        if fetch_context:
            context_parts.append(fetch_context)
        if vector_context:
            context_parts.append(vector_context)
        if feedback_context:
            context_parts.append(feedback_context)
        if summaries_text:
            context_parts.append(summaries_text)

        if not context_parts:
            return "No relevant content found in documents or project files.", None

        raw = "\n\n---\n\n".join(context_parts)
        logger.info(f"[explore] Returning raw context ({len(raw)} chars, {len(context_parts)} source(s))")
        return raw, citations

    def explore_and_read(self, user_message: str, cwd: str):
        """Explore -> pick best file -> read -> return content."""
        from smartloop.agents.orchestrator import MCPExecutionResult

        logger.info(f"[explore] Running explore → read pipeline in {cwd}")
        tool_names = []

        candidates = []
        for root, dirs, files in os.walk(cwd):
            dirs[:] = [
                d for d in dirs
                if d not in EXPLORE_SKIP_DIRS
                and not d.endswith(".egg-info")
            ]
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in EXPLORE_EXTENSIONS:
                    continue
                rel = os.path.relpath(os.path.join(root, fname), cwd)
                candidates.append(rel)
                if len(candidates) >= MAX_EXPLORE_FILES:
                    break
            if len(candidates) >= MAX_EXPLORE_FILES:
                break

        if not candidates:
            logger.info("[explore] No source files found")
            return MCPExecutionResult(handled=False)

        tool_names.append(EXPLORE_TOOL)
        file_list = "\n".join(f"  {f}" for f in candidates)
        logger.info(f"[explore] Found {len(candidates)} files")

        pick_prompt = (
            f"User question: {user_message}\n\n"
            f"Files in the project directory:\n{file_list}\n\n"
            "Which files are most relevant to answer the user's question? "
            "Pick up to 3 files. Reply with ONLY the file paths, one per line."
        )
        try:
            pick_response = str(self.llm.invoke(pick_prompt)).strip()
            logger.info(f"[explore] LLM picked files: {pick_response}")
        except Exception as e:
            logger.warning(f"[explore] File pick failed: {e}")
            pick_response = ""

        picked = []
        for line in pick_response.split("\n"):
            line = line.strip().lstrip("- ").strip("`")
            if line in candidates:
                picked.append(line)
        if not picked:
            for c in candidates:
                if "readme" in c.lower():
                    picked.append(c)
                    break
            if not picked:
                picked = candidates[:1]

        logger.info(f"[explore] Reading {len(picked)} file(s): {picked}")

        file_contents = []
        for fpath in picked[:3]:
            full_path = os.path.join(cwd, fpath)
            ext = os.path.splitext(fpath)[1].lower()
            try:
                from .file_ops import FileOperations
                if ext in FileOperations.BINARY_EXTENSIONS:
                    fo = FileOperations(self.llm)
                    content = fo.parse_binary_file(full_path)
                    if content is None:
                        logger.warning(f"[explore] Could not parse binary file {fpath}")
                        continue
                else:
                    content = open(full_path, "r", encoding="utf-8", errors="replace").read()
                if len(content) > MAX_FILE_CHARS:
                    content = content[:MAX_FILE_CHARS] + "\n... (truncated)"
                file_contents.append(f"--- {fpath} ---\n{content}")
                logger.info(f"[explore] Read {fpath} ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"[explore] Could not read {fpath}: {e}")
        tool_names.append(FILE_READ_TOOL)

        if not file_contents:
            return MCPExecutionResult(handled=False)

        # Return raw file contents; the final streaming LLM synthesizes.
        combined = "\n\n".join(file_contents)
        logger.info(f"[explore] Returning raw file contents ({len(combined)} chars)")

        return MCPExecutionResult(
            handled=True,
            content=combined,
            tool_name=FILE_READ_TOOL,
            server_name=SYSTEM_TOOLS_SERVER_ID,
            tool_names=tool_names,
        )

    # ================================================================== #
    # URL fetch + fetch step resolution                                    #
    # ================================================================== #

    def fetch_url(
        self, url: str, mcp_servers: Optional[List["MCPServer"]] = None,
    ) -> str:
        """Fetch a URL as markdown via the in-process FetchAgent.

        ``mcp_servers`` is accepted for backward compatibility but unused —
        fetching is no longer an MCP tool call.
        """
        logger.info(f"[explore] Fetching URL via FetchAgent: {url}")
        return self.fetch_agent.fetch(url)

    def resolve_fetch_steps(
        self,
        steps: List["PlanStep"],
        available_servers: List[Dict[str, Any]],
        user_message: str,
    ) -> List["PlanStep"]:
        _SYSTEM_TOOLS = {
            FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL, BASH_TOOL, EXPLORE_TOOL,
        }
        resolved = []
        for step in steps:
            # Built-in system tools are explicit MCP steps — never rewrite
            # them into a fetch step even when the request mentions a URL.
            if step.step_type == StepType.TOOL and step.tool_name not in _SYSTEM_TOOLS:
                instr_lower = step.instruction.lower()
                tool_lower = (step.tool_name or "").lower()
                _FETCH_INTENT_WORDS = {"fetch", "browse", "scrape", "navigate", "retrieve", "get url", "http"}
                is_fetch_intended = (
                    any(kw in tool_lower for kw in FETCH_TOOL_KEYWORDS)
                    or any(kw in instr_lower for kw in _FETCH_INTENT_WORDS)
                    or "url" in instr_lower
                )
                if is_fetch_intended:
                    urls = self.arg_extractor._extract_urls_from_text(user_message)
                    if not urls:
                        urls = self.arg_extractor._extract_urls_from_text(step.instruction)
                    if urls:
                        logger.info(
                            f"[resolve] Step {step.step}: URL '{urls[0]}' → FetchAgent step"
                        )
                        resolved.append(PlanStep(
                            step=step.step,
                            step_type=StepType.FETCH,
                            tool_name="fetch",
                            server_id="",
                            server_name="",
                            instruction=f"Fetch {urls[0]} and extract its main content. Context: {step.instruction}",
                            default_args={"url": urls[0]},
                        ))
                        continue
            resolved.append(step)
        return resolved
