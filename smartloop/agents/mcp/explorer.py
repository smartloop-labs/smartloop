"""
Explorer — unified system tool for exploring skills, vector search,
conversation history/context, filesystem walk, and URL fetch.
"""
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from smartloop.llms import BaseLLM
from smartloop.workspace.workspace import MCPServerType

from smartloop.agents.planner import PlanStep, StepType, MCPToolContext, SYSTEM_TOOLS_SERVER_ID

from .server_selector import (
    EXPLORE_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL, BASH_TOOL,
)
from .mcp_tool_caller import MCPToolCaller
from .argument_extractor import MCPArgumentExtractor

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
    """Collected search results from vector, feedback, workspace, and skill tools."""

    vector_context: str = ""
    feedback_context: str = ""
    workspace_context: str = ""
    citations: Optional[List[Dict[str, Any]]] = None
    skill_tool_results: List[Dict[str, Any]] = Field(default_factory=list)
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
            or self.skill_tool_results
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

    # ================================================================== #
    # Skill matching                                                       #
    # ================================================================== #

    def match_skills(
        self,
        user_message: str,
        parsed_skills: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        matchable = [
            s for s in parsed_skills
            if not s["name"].startswith("__")
        ]
        if not matchable:
            return []

        skill_list = "\n".join(
            f"  {i+1}. {s['name']}: {s['description']}"
            for i, s in enumerate(matchable)
        )

        prompt = (
            "<system>You are matching a user message to instruction sets (skills). "
            "Each skill describes when it applies and contains instructions for "
            "how to handle that type of user request. "
            "A user message may match multiple skills if the request spans "
            "several concerns. Pick ALL skills whose descriptions are relevant "
            "to the user's query. "
            "Reply with ONLY the number(s) of matching skills (comma-separated), "
            "or NONE if no skill is relevant.\n\n"
            f"Available skills:\n{skill_list}\n</system>"
            f"<user>{user_message}</user>"
        )

        try:
            response = str(self.llm.invoke(prompt)).strip().lower()
            logger.info(f"[skill-match] LLM response: {response[:100]}")

            if "none" in response and len(response) < 10:
                return []

            matched: List[Dict[str, str]] = []
            seen: set = set()

            # Parse numbers from the response (handles "1", "1, 3", "1,2")
            for token in re.findall(r'\d+', response):
                idx = int(token) - 1  # 1-based in prompt
                if 0 <= idx < len(matchable):
                    s = matchable[idx]
                    if s["name"] not in seen:
                        matched.append(s)
                        seen.add(s["name"])

            # Fallback: check if the LLM returned skill names instead
            if not matched:
                for s in matchable:
                    if s["name"].lower() in response and s["name"] not in seen:
                        matched.append(s)
                        seen.add(s["name"])

            if matched:
                logger.info(
                    f"[skill-match] Matched {len(matched)} skill(s): "
                    f"{[s['name'] for s in matched]}"
                )
                return matched

            # Fallback: keyword overlap between user message and skill
            # description.  Small models often return "none" even when
            # the skill clearly applies.  We tokenise both strings and
            # require ≥2 overlapping content words (length ≥ 4).
            matched = self._keyword_fallback_match(user_message, matchable)
            if matched:
                logger.info(
                    f"[skill-match] Matched {len(matched)} skill(s) via keyword fallback: "
                    f"{[s['name'] for s in matched]}"
                )
            return matched

        except Exception as e:
            logger.warning(f"[skill-match] LLM failed: {e}")

        # Even if LLM fails entirely, try keyword fallback
        return self._keyword_fallback_match(user_message, matchable)

    @staticmethod
    def _keyword_fallback_match(
        user_message: str,
        skills: List[Dict[str, str]],
        min_overlap: int = 2,
        min_word_len: int = 4,
    ) -> List[Dict[str, str]]:
        """Match skills by keyword overlap when LLM matching fails.

        Tokenises the user message and each skill's name + description,
        then checks for ≥ ``min_overlap`` shared content words (words
        with length ≥ ``min_word_len``).  This is language-agnostic —
        it works on raw word forms without stop-word lists.

        The overlap threshold increases with the number of skills to
        reduce false positives: threshold = min_overlap + (num_skills // 5).
        """
        num_skills = len(skills)
        # Scale threshold: more skills → require more keyword overlap
        # to reduce false-positive matches as the skill set grows.
        effective_threshold = min_overlap + (num_skills // 5)

        def _tokens(text: str) -> set:
            return {
                w for w in re.findall(r'\w+', text.lower())
                if len(w) >= min_word_len
            }

        msg_tokens = _tokens(user_message)
        if not msg_tokens:
            return []

        matched: List[Dict[str, str]] = []
        for s in skills:
            skill_text = f"{s.get('name', '')} {s.get('description', '')} {s.get('content', '')}"
            skill_tokens = _tokens(skill_text)
            overlap = msg_tokens & skill_tokens

            # Reject if the message is too short for keyword matching to
            # be reliable.  When the entire message overlaps (ratio == 1.0)
            # but has ≤ 3 tokens, it's almost certainly a false positive
            # from generic words like "proof read".
            if len(msg_tokens) <= 3 and len(overlap) == len(msg_tokens):
                continue

            if len(overlap) >= effective_threshold:
                logger.debug(
                    f"[skill-match] keyword overlap for '{s['name']}': {overlap}"
                )
                matched.append(s)

        return matched

    def extract_tool_names(
        self,
        matched_skills: List[Dict[str, str]],
        mcp_tool_contexts: List[MCPToolContext],
    ) -> List[str]:
        available_tools: set = set()
        for srv in mcp_tool_contexts:
            for tool in srv.tools:
                name = tool.get("name", "")
                if name:
                    available_tools.add(name)

        referenced: List[str] = []
        seen: set = set()

        for sk in matched_skills:
            content = sk.get("content", "")
            for match in re.findall(r'`([^`]+)`|\*\*([^*]+)\*\*', content):
                name = (match[0] or match[1]).strip()
                if name in available_tools and name not in seen:
                    referenced.append(name)
                    seen.add(name)

            # Also match plain-text mentions of known tool names in content
            for tool_name in available_tools:
                if tool_name not in seen and tool_name in content:
                    referenced.append(tool_name)
                    seen.add(tool_name)

            # Extract tool names from ```mcp blocks
            for m in re.finditer(r'```mcp\s*\n\s*\w+\.(\w+)\(', content):
                name = m.group(1)
                if name in available_tools and name not in seen:
                    referenced.append(name)
                    seen.add(name)

        return referenced

    # ================================================================== #
    # Collect: unified search + skill tool execution                       #
    # ================================================================== #

    def collect(
        self,
        query: str,
        model_service: Optional[Any] = None,
        cwd: Optional[str] = None,
        enriched: Optional[EnrichedContext] = None,
        matched_skills: Optional[List[Dict]] = None,
        mcp_servers: Optional[List["MCPServer"]] = None,
        mcp_tool_contexts: Optional[List[MCPToolContext]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> SearchContext:
        """Collect all context: vector search, feedback, workspace, and skill tool calls.

        Uses enriched context (from enrich_light) to improve skill tool argument
        extraction — history insight and MCP catalog inform better tool execution.
        """
        ctx = SearchContext()

        if not model_service:
            return ctx

        # 1. Vector/document search
        try:
            doc_result = model_service.get_document_context_with_citations(
                query=query,
                conversation_history=conversation_history,
            )
            if doc_result is not None:
                ctx.vector_context = doc_result[0]
                ctx.citations = doc_result[1]
                logger.info(
                    f"[collect] Vector search returned {len(ctx.vector_context)} chars"
                    f" from {len(ctx.citations)} document(s)"
                )
            else:
                logger.info("[collect] Vector search returned no results")
        except Exception as e:
            if isinstance(e, _PROGRAMMING_ERRORS):
                raise
            logger.warning(f"[collect] Vector search failed: {e}")

        # 2. Feedback search
        try:
            feedback_chunks = model_service.search_feedback(query, n_results=5)
            if feedback_chunks:
                ctx.feedback_context = "\n---\n".join(c.text for c in feedback_chunks)
                logger.info(
                    f"[collect] Feedback search returned {len(feedback_chunks)} Q&A pair(s)"
                )
        except Exception as e:
            if isinstance(e, _PROGRAMMING_ERRORS):
                raise
            logger.warning(f"[collect] Feedback search failed: {e}")

        # 3. Workspace search with dedup
        if cwd:
            try:
                ws_chunks = model_service.search_workspace(query, cwd, n_results=10)
                if ws_chunks:
                    seen_files: dict = {}
                    for chunk in ws_chunks:
                        fname = chunk.document_name or chunk.document_id
                        # Skip skill files — they are instructions, not data
                        if fname and (
                            "/skills/" in fname
                            or fname.startswith("skills/")
                            or fname.upper().endswith("SKILL.MD")
                        ):
                            continue
                        if fname not in seen_files or chunk.distance < seen_files[fname].distance:
                            seen_files[fname] = chunk
                    ws_texts = []
                    for fname, chunk in seen_files.items():
                        ws_texts.append(f"--- {fname} ---\n{chunk.text[:2000]}")
                    ctx.workspace_context = "\n".join(ws_texts)
                    logger.info(
                        f"[collect] Workspace search returned {len(seen_files)} file(s)"
                    )
            except Exception as e:
                if isinstance(e, _PROGRAMMING_ERRORS):
                    raise
                logger.warning(f"[collect] Workspace search failed: {e}")

        # 4. Execute skill-referenced tools
        if matched_skills and mcp_tool_contexts and mcp_servers:
            skill_tool_names = self.extract_tool_names(matched_skills, mcp_tool_contexts)
            for tool_name in skill_tool_names:
                result = self._execute_skill_tool(
                    tool_name, query, mcp_servers, mcp_tool_contexts,
                    enriched=enriched,
                )
                if result:
                    ctx.skill_tool_results.append(result)

        return ctx

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

    @staticmethod
    def _search_workspace(query: str, model_service: Any, cwd: str) -> str:
        """Workspace file search — thread-safe, I/O bound."""
        try:
            ws_chunks = model_service.search_workspace(query, cwd, n_results=10)
            if ws_chunks:
                seen_files: dict = {}
                for chunk in ws_chunks:
                    fname = chunk.document_name or chunk.document_id
                    if fname and (
                        "/skills/" in fname
                        or fname.startswith("skills/")
                        or fname.upper().endswith("SKILL.MD")
                        or fname.startswith("mcp-")
                    ):
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
    ) -> SearchContext:
        """Fan-out vector/feedback/workspace searches in parallel.

        Drop-in replacement for :meth:`collect` — same signature (plus
        optional ``executor``).  The three I/O-bound searches run
        concurrently via a :class:`ThreadPoolExecutor`.  Skill tool
        execution still runs sequentially afterward (tools may depend
        on search results).
        """
        ctx = SearchContext()

        if not model_service:
            return ctx

        pool = executor or ThreadPoolExecutor(max_workers=3)
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
            ) if cwd else None

            # Collect results
            vec_result = f_vec.result()
            if vec_result is not None:
                ctx.vector_context = vec_result[0]
                ctx.citations = vec_result[1]
                ctx.documents = vec_result[2]

            ctx.feedback_context = f_fb.result()

            if f_ws is not None:
                ctx.workspace_context = f_ws.result()

        finally:
            if own_pool:
                pool.shutdown(wait=False)

        # Skill tool execution runs in the DAG step nodes — do NOT run them
        # here during collection.  Eager execution here would bypass argument
        # extraction, validation, and result accumulation in reason().

        return ctx

    def _execute_skill_tool(
        self,
        tool_name: str,
        query: str,
        mcp_servers: List["MCPServer"],
        mcp_tool_contexts: List[MCPToolContext],
        enriched: Optional[EnrichedContext] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a skill-referenced tool from servers, extract args, call it, return result."""
        if not self.tool_caller or not self.arg_extractor:
            return None

        # Find the tool and its server — first via planning contexts, then
        # fall back to searching mcp_servers directly (system servers like
        # __fetch__ are excluded from mcp_tool_contexts).
        resolved_server = None
        tool_schema = None
        for srv_ctx in mcp_tool_contexts:
            for tool in srv_ctx.tools:
                if tool.get("name", "") == tool_name:
                    for srv in mcp_servers:
                        if srv.id == srv_ctx.server_id:
                            resolved_server = srv
                            tool_schema = tool
                            break
                    break
            if resolved_server:
                break

        if not resolved_server:
            for srv in mcp_servers:
                if not srv.enabled or not srv.tools:
                    continue
                for t in srv.tools:
                    if t.name == tool_name:
                        resolved_server = srv
                        tool_schema = {
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.input_schema,
                        }
                        break
                if resolved_server:
                    break

        if not resolved_server or not tool_schema:
            logger.info(f"[collect] Skill tool '{tool_name}' not found in servers")
            return None

        # Skip fetch tools if no URL in query
        if any(kw in tool_name.lower() for kw in FETCH_TOOL_KEYWORDS):
            if not re.search(r'https?://[^\s<>\'")\]]+', query):
                logger.info(
                    f"[collect] Skill tool '{tool_name}' is a fetch tool "
                    f"but no URL in query — skipping"
                )
                return None

        input_schema = tool_schema.get("input_schema", {}) or {}

        # Build extraction context with enriched info for better arg extraction
        extraction_input = query
        if enriched and enriched.history_insight:
            extraction_input = f"{query}\n\nContext: {enriched.history_insight}"

        try:
            arguments = self.arg_extractor.invoke(
                user_input=extraction_input,
                tool_name=tool_name,
                input_schema=input_schema,
            )
            arguments.pop("_missing_params", None)
        except Exception as e:
            logger.warning(f"[collect] Arg extraction failed for '{tool_name}': {e}")
            return None

        access_token = self.tool_caller.get_access_token(resolved_server)

        try:
            result = self.tool_caller.call_tool(
                server=resolved_server,
                tool_name=tool_name,
                arguments=arguments,
                access_token=access_token,
            )
        except Exception as e:
            logger.warning(f"[collect] Tool call failed for '{tool_name}': {e}")
            return None

        if not result.success or not result.content:
            logger.warning(f"[collect] Skill tool '{tool_name}' returned no result")
            return None

        raw_parts = [
            block.get("text", "") if isinstance(block, dict) and block.get("type") == "text"
            else str(block)
            for block in result.content
        ]
        raw = "\n".join(raw_parts)
        logger.info(f"[collect] Skill tool '{tool_name}' returned {len(raw)} chars")

        return {
            "tool_name": tool_name,
            "content": raw,
            "server_name": resolved_server.name,
        }

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
        """Format pre-filtered history turns into a concise text block.

        The history is already relevance-filtered by the caller (via
        ``ConversationStore.query_relevant``), so this method simply
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
            pool = executor or ThreadPoolExecutor(max_workers=4)
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
        self, url: str, mcp_servers: List["MCPServer"],
    ) -> str:
        sorted_servers = sorted(
            mcp_servers,
            key=lambda s: 0 if s.server_type == MCPServerType.LOCAL else 1,
        )
        for server in sorted_servers:
            if not server.enabled or not server.tools:
                continue
            for tool in server.tools:
                t_name = (tool.name or "").lower()
                t_desc = (tool.description or "").lower()
                if not any(kw in t_name or kw in t_desc for kw in FETCH_TOOL_KEYWORDS):
                    continue

                url_param = None
                props = (tool.input_schema or {}).get("properties", {})
                for p_name, p_info in props.items():
                    if self.arg_extractor._is_url_parameter(p_name, p_info):
                        url_param = p_name
                        break
                url_param = url_param or "url"

                logger.info(f"[explore] Fetching URL via '{tool.name}': {url}")
                try:
                    result = self.tool_caller.call_tool(
                        server=server,
                        tool_name=tool.name,
                        arguments={url_param: url},
                        access_token=self.tool_caller.get_access_token(server),
                    )
                    if not result.success or not result.content:
                        logger.warning(f"[explore] Fetch failed: {result.error}")
                        return ""

                    raw_parts = [
                        block.get("text", "") if isinstance(block, dict) and block.get("type") == "text"
                        else str(block)
                        for block in result.content
                    ]
                    raw = "\n".join(raw_parts)

                    if self.is_javascript_page(raw):
                        msg = (
                            f"The page at {url} requires JavaScript to render its content. "
                            "JavaScript-rendered pages cannot be processed."
                        )
                        logger.info(f"[explore] JavaScript page detected: {url}")
                        return msg

                    logger.info(f"[explore] Fetched {len(raw)} chars from {url}")
                    return raw
                except Exception as e:
                    logger.warning(f"[explore] Fetch error for {url}: {e}")
                    return ""

        logger.info("[explore] No fetch-capable MCP tool found")
        return ""

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
                        fetch_step = self._make_fetch_step(urls[0], step.instruction, available_servers)
                        if fetch_step:
                            logger.info(
                                f"[resolve] Step {step.step}: URL '{urls[0]}' → "
                                f"deterministic fetch via '{fetch_step.server_name}/{fetch_step.tool_name}'"
                            )
                            resolved.append(PlanStep(
                                step=step.step,
                                step_type=StepType.TOOL,
                                tool_name=fetch_step.tool_name,
                                server_id=fetch_step.server_id,
                                server_name=fetch_step.server_name,
                                instruction=fetch_step.instruction,
                            ))
                            continue
            resolved.append(step)
        return resolved

    def _make_fetch_step(self, url: str, instruction: str, available_servers: List[Dict[str, Any]]) -> Optional["PlanStep"]:
        sorted_servers = sorted(
            available_servers,
            key=lambda s: 0 if s.get("server_type") == MCPServerType.LOCAL else 1,
        )
        for srv in sorted_servers:
            if srv.get("id") == SYSTEM_TOOLS_SERVER_ID:
                continue
            for tool in srv.get("tools", []):
                t_name = tool.get("name", "").lower()
                t_desc = tool.get("description", "").lower()
                if any(kw in t_name or kw in t_desc for kw in FETCH_TOOL_KEYWORDS):
                    return PlanStep(
                        step=0,
                        step_type=StepType.TOOL,
                        tool_name=tool["name"],
                        server_id=srv["id"],
                        server_name=srv.get("name", ""),
                        instruction=f"Fetch {url} and extract its main content. Context: {instruction}",
                    )
        return None

    @staticmethod
    def is_javascript_page(content: str) -> bool:
        if not content:
            return False
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        noscript = soup.find("noscript")
        if noscript:
            noscript_text = noscript.get_text(strip=True).lower()
            js_phrases = [
                "enable javascript", "javascript is required",
                "javascript is needed", "requires javascript",
                "turn on javascript", "javascript must be enabled",
                "please enable javascript",
            ]
            if any(phrase in noscript_text for phrase in js_phrases):
                return True
        body = soup.find("body") or soup
        visible_text = body.get_text(separator=" ", strip=True)
        scripts = soup.find_all("script")
        if len(scripts) > 0 and len(visible_text) < 200:
            return True
        root_div = soup.find("div", id=re.compile(r"^(root|app|__next)$"))
        if root_div and not root_div.get_text(strip=True):
            return True
        return False
