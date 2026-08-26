"""Running one planned step: resolve arguments, then call.

The step types split two ways.  A TOOL step needs its arguments worked out
first — which server, which tool, what values — and :meth:`resolve_tool_args`
does that before :meth:`call_tool` runs it.  Every other type (explore, fetch,
conversational, action) executes directly in :meth:`execute`.

Argument resolution is the hard half and most of the volume here: values have
to come from the user's message and upstream results, never from the tool's
own description, or a small model copies the schema back as data.
"""
import logging
import os
import re
from typing import Any, Dict, List, Optional

from smartloop.llms import BaseLLM
from smartloop.workspace.workspace import MCPServer, MCPServerType
from smartloop.agents.nodes import NodeResult
from smartloop.agents.mcp.argument_extractor import MCPArgumentExtractor
from smartloop.agents.mcp.dependency_analyzer import MCPDependencyAnalyzer
from smartloop.agents.mcp.explorer import Explorer
from smartloop.agents.mcp.mcp_agent import MCPAgent
from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
from smartloop.agents.mcp.server_selector import (
    BASH_TOOL, EXPLORE_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL,
)
from smartloop.agents.orchestration.context import _OrchestratorContext
from smartloop.agents.orchestration.response_builder import ResponseBuilder
from smartloop.agents.planner import (
    PlanStep, StepType, SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME,
)
from smartloop.utils.mcp_result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


def plan_step_result(
    node_id: str,
    step: PlanStep,
    content: str,
    raw: Optional[str] = None,
) -> NodeResult:
    """Build the NodeResult for one executed plan step.

    Each node stores its own output — the next node reads only its direct
    dependency's ``raw`` field, which is what keeps the context focused
    instead of dragging the whole accumulated chain forward.
    """
    tool_name = step.tool_name or step.step_type.value
    raw_value = raw or content
    return NodeResult(node_id=node_id, success=True, data={
        "tool_name": tool_name,
        "content": content,
        "raw": raw_value,
        "is_tool_result": step.step_type == StepType.TOOL,
    })


class StepExecutor:
    """Resolves and runs individual plan steps."""

    def __init__(
        self,
        orchestrator_llm: BaseLLM,
        tool_caller: MCPToolCaller,
        arg_extractor: MCPArgumentExtractor,
        dep_analyzer: MCPDependencyAnalyzer,
        result_formatter: ResultFormatter,
        explorer: Explorer,
        mcp_agent: MCPAgent,
        response: ResponseBuilder,
    ):
        self.orchestrator_llm = orchestrator_llm
        self.tool_caller = tool_caller
        self.arg_extractor = arg_extractor
        self.dep_analyzer = dep_analyzer
        self.result_formatter = result_formatter
        self.explorer = explorer
        self._mcp_agent = mcp_agent
        self._response = response

    def call_tool_for_server(self, *args, **kwargs):
        return self.tool_caller.call_tool(*args, **kwargs)

    def evaluate_condition(self, condition: str, previous_results: List[Dict]) -> bool:
        """Evaluate a step condition against previous results.

        Supported formats:
        - ``result_contains:<text>`` — case-insensitive substring check on
          any previous result text
        - ``result_contains_any:<w1,w2,w3>`` — true if ANY keyword matches
        - ``llm:<question>`` — sends previous results + question to LLM,
          expects YES/NO
        - ``None`` / empty — returns ``True`` (always execute)
        """
        if not condition:
            return True

        condition = condition.strip()

        # result_contains:<text>
        if condition.lower().startswith("result_contains:"):
            needle = condition[len("result_contains:"):].strip().lower()
            for r in previous_results:
                text = str(r.get("result", "") or r.get("content", ""))
                if needle in text.lower():
                    return True
            return False

        # llm:<condition clause>
        if condition.lower().startswith("llm:"):
            clause = condition[4:].strip()
            # Build a short summary of previous results
            combined = ""
            for r in previous_results:
                text = str(r.get("result", "") or r.get("content", ""))
                combined += text[:1500] + "\n"
            combined = combined.strip()[:3000]

            if not combined:
                logger.info("[condition] No previous results — defaulting to YES")
                return True

            prompt = (
                f"Results:\n{combined}\n\n"
                f"Condition: {clause}\n"
                "YES or NO?"
            )
            try:
                response = str(self.orchestrator_llm.invoke(
                    prompt, max_tokens=DECISION_MAX_TOKENS,
                )).strip().upper()
                # Fail-open: only skip if the LLM explicitly says NO
                # without any YES. This handles ambiguous responses by
                # defaulting to execute.
                has_no = "NO" in response
                has_yes = ("YES" in response or response.startswith("Y")
                           or "TRUE" in response)
                result = has_yes or not has_no
                logger.info(
                    f"[condition] LLM: '{response[:80]}' → {result}"
                )
                return result
            except Exception as e:
                logger.warning(f"[condition] LLM evaluation failed: {e}")
                return True  # fail-open

        # Unknown format — execute unconditionally
        return True

    @staticmethod
    def group_independent_steps(
        steps: List[PlanStep],
    ) -> List[List[PlanStep]]:
        """Group plan steps into batches that can run in parallel.

        A step is independent when it has no ``condition``, no explicit
        dependency on previous results, and is not inherently sequential
        (e.g. write/summarize actions depend on prior context).

        Consecutive independent steps are grouped together; steps with
        conditions, dependencies, or inherently-sequential types form
        singleton groups that flush any accumulated parallel batch.
        """
        _DEP_RE = re.compile(r'\bstep\s*\d|\bprevious\b|\bresult\s*of\b', re.IGNORECASE)

        # Action kinds that inherently depend on prior results
        _SEQUENTIAL_ACTIONS = {"write", "summarize"}

        groups: List[List[PlanStep]] = []
        current_parallel: List[PlanStep] = []

        for step in steps:
            action_kind = getattr(step, "action_kind", None) or ""
            is_independent = (
                not step.condition
                and not _DEP_RE.search(step.instruction or "")
                and action_kind not in _SEQUENTIAL_ACTIONS
                and step.step_type != StepType.CONVERSATIONAL
            )
            if is_independent:
                current_parallel.append(step)
            else:
                # Flush any accumulated parallel group
                if current_parallel:
                    groups.append(current_parallel)
                    current_parallel = []
                groups.append([step])

        if current_parallel:
            groups.append(current_parallel)

        return groups

    def execute(
        self,
        ctx: _OrchestratorContext,
        step: PlanStep,
        previous_results: Optional[List[Dict[str, Any]]] = None,
        document_context: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
    ) -> Optional[tuple]:
        """Run one non-TOOL step and return ``(formatted, raw)`` or None.

        Everything constant for the request comes off ``ctx``; only what
        varies per step (the step itself, its upstream results, the document
        context in force, and the resolved model) is passed in.
        """
        # Unpacked once rather than read through ``ctx`` throughout: the body
        # below is long and branches per step type, and the plain names keep
        # the individual branches readable.
        user_message = ctx.user_message
        mcp_servers = ctx.mcp_servers
        available_servers = ctx.available_servers
        cwd = ctx.effective_cwd
        project_dir = ctx.project_dir
        model_service = ctx.model_service
        conversation_history = ctx.conversation_history

        tool_name = step.tool_name or ""
        has_explicit_tool = bool(step.tool_name)
        directive = step.instruction or user_message
        work_dir = os.path.expanduser(cwd or project_dir or os.getcwd())

        logger.info(
            f"[step] {step.server_name}/{tool_name}, cwd={work_dir}, "
            f"prior_results={len(previous_results or [])}"
        )

        step_type = step.step_type

        # -- ANALYZE --
        if step_type == StepType.ANALYZE:
            logger.warning("[step] analyze step — no enriched analysis available")
            return None

        # -- CONVERSATIONAL --
        # When a capability model was resolved, generate the response here
        # so the router can stream the final content directly.  Otherwise
        # fall back to collecting raw context for the router's LLM.
        if step_type == StepType.CONVERSATIONAL:
            parts: List[str] = []
            if previous_results:
                for r in previous_results:
                    parts.append(f"[{r['tool_name']}]: {r['result']}")
            if document_context and "DOCUMENT CONTEXT" not in directive:
                parts.append(f"DOCUMENT CONTEXT:\n{document_context}")
            if directive:
                parts.append(directive)
            raw = "\n\n".join(parts) if parts else directive

            if llm is not None and step.capability:
                from smartloop.constants import SYSTEM_PROMPT, resolve_current_date
                # The date is runtime grounding the model wasn't trained on, so
                # it rides with the user turn (next to the request) instead of
                # the system prompt where a small model is unlikely to use it.
                date_str = ctx.current_date or resolve_current_date()
                # Include the session's prior turns so a follow-up like
                # "the top place from that list" can reference the assistant's
                # earlier reply instead of claiming nothing was provided.
                history_block = ""
                for turn in conversation_history or []:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if role in ("user", "assistant") and content:
                        history_block += f"<{role}>{content}</{role}>"
                user_block = f"{raw}\n\n[Current date: {date_str}]"
                prompt = f"<system>{SYSTEM_PROMPT}</system>{history_block}<user>{user_block}</user>"
                try:
                    response = str(llm.invoke(prompt)).strip()
                    if response:
                        logger.info(
                            f"[step] conversational generated {len(response)} chars "
                            f"via capability={step.capability}"
                        )
                        return response, response
                except Exception as e:
                    logger.warning(
                        f"[step] conversational LLM invoke failed: {e} — "
                        f"falling back to raw context"
                    )

            logger.info(f"[step] conversational collected {len(raw)} chars of raw context")
            return raw, raw

        # -- EXPLORE --
        if step_type == StepType.EXPLORE:
            try:
                content, explore_citations = self.explorer.explore(
                    user_message, work_dir,
                    project_dir=project_dir, mcp_servers=mcp_servers,
                    model_service=model_service,
                    vector_context=document_context or "",
                    enriched=None,
                )
            except Exception as e:
                logger.error(f"[step] explore failed: {e}")
                return None
            if not content:
                return None
            if explore_citations:
                ctx.explore_citations = explore_citations
            return content, content

        # -- ACTION --
        # Only the internal ``summarize`` kind remains here. The built-in
        # file/grep/bash tools now run as TOOL steps through the smartloop
        # MCP server (resolved/executed via the resolve+execute handlers).
        if step_type == StepType.ACTION:
            action_kind = step.action_kind or ""

            if action_kind == "summarize":
                # Collect raw context; the final streaming LLM handles reasoning.
                parts = []
                if previous_results:
                    for r in previous_results:
                        parts.append(f"[{r['tool_name']}]: {r['result']}")
                if directive:
                    parts.append(directive)
                raw = "\n\n".join(parts) if parts else directive
                logger.info(f"[step] summarize collected {len(raw)} chars of raw context")
                return raw, raw

            logger.warning(f"[step] Unknown action_kind: {action_kind}")
            return None

        # -- TOOL: Call external MCP tool --
        server = None
        if step.server_id:
            server = self.tool_caller.find_server(mcp_servers, step.server_id, step.server_name)
        if not server:
            # Catalogue scan: find the server that owns this tool
            real_server_id = None
            for srv in available_servers:
                if srv.get("id") == SYSTEM_TOOLS_SERVER_ID:
                    continue
                for t in srv.get("tools", []):
                    if t.get("name") == tool_name:
                        real_server_id = srv["id"]
                        logger.info(
                            f"[step] Found tool '{tool_name}' in server "
                            f"'{srv.get('name', '')}' (id={srv['id']})"
                        )
                        break
                if real_server_id:
                    break

            # Fuzzy server name match (when tool not found by exact name)
            if not real_server_id and step.server_name:
                for srv in available_servers:
                    if srv.get("id") == SYSTEM_TOOLS_SERVER_ID:
                        continue
                    srv_name = srv.get("name", "")
                    if (srv_name.lower() == step.server_name.lower()
                            or step.server_name.lower() in srv_name.lower()):
                        real_server_id = srv["id"]
                        logger.info(
                            f"[step] Matched server '{srv_name}' via "
                            f"name hint '{step.server_name}'"
                        )
                        break

            # Fuzzy: check directive for server name hints
            if not real_server_id:
                for srv in available_servers:
                    if srv.get("id") == SYSTEM_TOOLS_SERVER_ID:
                        continue
                    srv_name = srv.get("name", "")
                    if srv_name and srv_name.lower() in directive.lower():
                        real_server_id = srv["id"]
                        logger.info(
                            f"[step] Matched server '{srv_name}' from directive"
                        )
                        break

            if real_server_id:
                server = self.tool_caller.find_server(mcp_servers, real_server_id, real_server_id)
                if server:
                    logger.info(
                        f"[step] Resolved server for tool '{tool_name}' "
                        f"via catalogue scan → '{server.name}' (id={server.id})"
                    )
                    if not self.tool_caller.find_tool_schema(server, tool_name):
                        if not has_explicit_tool:
                            matched_tool = self.tool_caller.find_tool_by_keyword(server, directive)
                            if matched_tool:
                                tool_name = matched_tool
                                logger.info(
                                    f"[step] Resolved tool '{step.tool_name}' → "
                                    f"'{tool_name}' via keyword match on server '{server.name}'"
                                )
                        else:
                            logger.warning(
                                f"[step] Explicit tool '{tool_name}' not found on "
                                f"server '{server.name}' — skipping keyword fallback"
                            )
        # Last resort: try every non-system server for the tool
        if not server:
            for srv_obj in mcp_servers:
                if not srv_obj.enabled or not srv_obj.tools:
                    continue
                if srv_obj.id == SYSTEM_TOOLS_SERVER_ID:
                    continue
                # Check exact match
                for t in srv_obj.tools:
                    if t.name == tool_name:
                        server = srv_obj
                        logger.info(
                            f"[step] Found tool '{tool_name}' via "
                            f"direct MCP server scan → '{srv_obj.name}'"
                        )
                        break
                # Check keyword match
                if not server and not has_explicit_tool:
                    matched_tool = self.tool_caller.find_tool_by_keyword(srv_obj, tool_name)
                    if matched_tool:
                        server = srv_obj
                        tool_name = matched_tool
                        logger.info(
                            f"[step] Resolved '{step.tool_name}' → '{tool_name}' "
                            f"via keyword on '{srv_obj.name}'"
                        )
                if server:
                    break

        if not server:
            logger.warning(f"[step] Server not found for tool '{tool_name}': {step.server_name} (id={step.server_id})")
            return None

        tool_schema = self.tool_caller.find_tool_schema(server, tool_name)
        if not tool_schema and not has_explicit_tool:
            # Fallback: LLM may have invented a tool name.  Try to resolve
            # the correct tool via keyword match from the step instruction.
            resolved_tool = self.tool_caller.find_tool_by_keyword(server, directive)
            if resolved_tool:
                logger.info(
                    f"[step] Tool '{tool_name}' not found — resolved to "
                    f"'{resolved_tool}' via keyword match on instruction"
                )
                tool_name = resolved_tool
                tool_schema = self.tool_caller.find_tool_schema(server, tool_name)

            # If still not found, check if the instruction mentions a known tool name
            if not tool_schema:
                for t in (server.tools or []):
                    t_name = t.name if hasattr(t, 'name') else t.get('name', '')
                    if t_name and t_name in directive:
                        logger.info(
                            f"[step] Tool '{tool_name}' not found — resolved to "
                            f"'{t_name}' (mentioned in instruction)"
                        )
                        tool_name = t_name
                        tool_schema = self.tool_caller.find_tool_schema(server, tool_name)
                        break

            if not tool_schema:
                logger.warning(f"[step] Tool schema not found: {tool_name} on server '{server.name}'")
                return None

        input_schema = tool_schema.get('input_schema', {}) or {}
        server_url = self.tool_caller.get_server_url(server)
        access_token = self.tool_caller.get_access_token(server)

        if not server_url and server.server_type != MCPServerType.LOCAL:
            logger.warning(f"[step] No server URL configured for: {step.server_name}")
            return None

        all_tools = [
            {'name': t.name, 'description': t.description, 'input_schema': t.input_schema}
            for t in server.tools
        ]
        dep_context = directive
        if previous_results:
            prior_text = "\n".join(
                f"[{r.get('tool_name', 'tool')}]: {str(r.get('result', ''))[:500]}"
                for r in previous_results
            )
            dep_context = f"{directive}\n\nPrevious step results:\n{prior_text}"

        dep_analysis = self.dep_analyzer.invoke(
            user_input=dep_context,
            tool_name=tool_name,
            input_schema=input_schema,
            available_tools=all_tools,
        )
        if dep_analysis.has_dependencies and not dep_analysis.can_execute:
            logger.info(f"[step] '{tool_name}' has unresolvable dependencies; falling through")
            return None

        resolved_ids = {}
        if dep_analysis.has_dependencies and dep_analysis.can_execute:
            resolved_ids = self.tool_caller.execute_prerequisites(
                dep_analysis=dep_analysis,
                server=server,
                user_message=directive,
                input_schema=input_schema,
                previous_results=previous_results,
            )

        # Use the user's original message as primary input so the LLM
        # extracts values from it, not from skill guidance metadata.
        # NOTE: Do NOT inject the tool description into the extraction
        # input. Small models copy it verbatim into same-named params
        # (e.g. a tool description "Create or update a Linear issue."
        # leaking into a `description` field). The extractor prompt
        # already names the tool; values must come from the user
        # message + previous results only.
        extraction_input = f"{user_message}"
        if previous_results:
            prior_parts = [
                str(r.get('result', ''))[:4000]
                for r in previous_results
            ]
            prior_data = "\n".join(prior_parts)
            extraction_input = (
                f"{user_message}\n\n"
                f"---\nData from previous step (use this to fill parameter values):\n"
                f"{prior_data}\n---"
            )

        # NOTE: Do NOT append step.instruction to extraction_input.
        # The instruction may contain skill body text (descriptions,
        # tool names, etc.) that small models pick up as parameter
        # values.  The tool schema already tells the LLM what
        # parameters are needed; the user message + previous results
        # provide the values.

        if resolved_ids:
            arguments = self.arg_extractor.invoke_with_resolved_ids(
                user_input=extraction_input,
                tool_name=tool_name,
                input_schema=input_schema,
                resolved_ids=resolved_ids,
                llm=llm,
            )
        else:
            arguments = self.arg_extractor.invoke(
                user_input=extraction_input,
                tool_name=tool_name,
                input_schema=input_schema,
                previous_tool_results=previous_results,
                llm=llm,
            )

        missing = arguments.pop('_missing_params', None)
        if missing:
            logger.info(f"[step] '{tool_name}' missing required params ({', '.join(missing)}); skipping")
            return None

        required_params = input_schema.get('required', [])
        properties = input_schema.get('properties', {})
        still_missing = []
        for param_name in required_params:
            val = arguments.get(param_name)
            if val is None or val == "" or val == "null":
                param_info = properties.get(param_name, {})
                # Only fill URL params from extracted URLs — never invent values
                if self.arg_extractor._is_url_parameter(param_name, param_info):
                    urls = self.arg_extractor._extract_urls_from_text(directive)
                    if not urls:
                        urls = self.arg_extractor._extract_urls_from_text(user_message)
                    if urls:
                        arguments[param_name] = urls[0]
                    else:
                        still_missing.append(param_name)
                else:
                    still_missing.append(param_name)
        if still_missing:
            logger.info(f"[step] '{tool_name}' still missing required params ({', '.join(still_missing)}); skipping")
            return None

        invalid = self.tool_caller.validate_argument_values(arguments, properties)
        if invalid:
            required_params = input_schema.get('required', [])
            hard_invalid = [msg for msg in invalid if any(
                f"'{p}'" in msg for p in required_params
            )]
            if hard_invalid:
                logger.info(f"[step] '{tool_name}' invalid required argument values: {hard_invalid}; falling through")
                return None
            for msg in invalid:
                for p in list(arguments):
                    if f"'{p}'" in msg and p not in required_params:
                        logger.info(f"[step] '{tool_name}' dropping invalid optional param '{p}': {msg}")
                        del arguments[p]

        if properties:
            unknown = [k for k in list(arguments) if k not in properties]
            if unknown:
                logger.info(f"[step] '{tool_name}' removing unknown params: {unknown}")
                for k in unknown:
                    del arguments[k]

        # Strip placeholder values the LLM generates for optional params
        # (e.g. "Not specified", "needs to be provided by the user").
        _PLACEHOLDER_RE = re.compile(
            r'not\s+(?:specified|provided|available|given)|'
            r'needs?\s+to\s+be\s+provided|'
            r'n/?a\b|unknown',
            re.IGNORECASE,
        )
        for k in list(arguments):
            val = arguments[k]
            if isinstance(val, str) and _PLACEHOLDER_RE.search(val):
                if k not in required_params:
                    logger.info(f"[step] '{tool_name}' dropping placeholder optional param '{k}': {val[:60]}")
                    del arguments[k]

        logger.info(f"[step] Calling '{tool_name}' with arguments: {arguments}")
        result = self.tool_caller.call_tool(
            server=server, tool_name=tool_name,
            arguments=arguments, access_token=access_token,
        )

        if not result.success and result.error and "401" in result.error and server_url:
            new_token = self.tool_caller.refresh_access_token(server)
            if new_token:
                result = self.tool_caller.call_tool(
                    server=server, tool_name=tool_name,
                    arguments=arguments, access_token=new_token,
                )

        if not result.success:
            logger.warning(f"[step] '{tool_name}' failed: {result.error}")
            return None
        if not result.content or result.is_error:
            error_detail = ""
            if result.content:
                raw = " ".join(
                    block.get('text', '') if isinstance(block, dict) else str(block)
                    for block in result.content
                ).strip()
                if raw:
                    error_detail = f": {raw[:300]}"
            logger.warning(f"[step] '{tool_name}' returned empty or error result{error_detail}")
            return None

        formatted_content = self.result_formatter.format_tool_result(
            tool_name=tool_name, content=result.content, is_error=result.is_error,
        )
        raw_parts = [
            block.get('text', '') if isinstance(block, dict) and block.get('type') == 'text'
            else str(block)
            for block in result.content
        ]
        raw_content = "\n".join(raw_parts)

        logger.info(
            f"[step] '{tool_name}' returned {len(formatted_content)} chars "
            f"({self.orchestrator_llm.get_num_tokens(formatted_content)} tokens)"
        )
        logger.info(f"[step] '{tool_name}' raw result: {raw_content[:1000]}")
        return formatted_content, raw_content

    def resolve_tool_args(
        self,
        ctx: _OrchestratorContext,
        step: PlanStep,
        previous_results: Optional[List[Dict[str, Any]]] = None,
        llm: Optional[BaseLLM] = None,
    ) -> Optional[tuple]:
        """Resolve tool arguments: find server, run deps, extract args.

        Returns ``(server, tool_name, arguments)`` or ``None`` if resolution fails.
        """
        # See _execute_step: unpacked for readability across a long body.
        user_message = ctx.user_message
        mcp_servers = ctx.mcp_servers
        available_servers = ctx.available_servers
        cwd = ctx.effective_cwd
        project_dir = ctx.project_dir

        tool_name = step.tool_name or ""
        has_explicit_tool = bool(step.tool_name)
        directive = step.instruction or user_message
        work_dir = os.path.expanduser(cwd or os.getcwd())

        logger.info(
            f"[resolve] {step.server_name}/{tool_name}, cwd={work_dir}, "
            f"prior_results={len(previous_results or [])}"
        )

        # Built-in tools live on the smartloop system server. If a plan
        # mis-assigned one to another server, pin it back to smartloop so it
        # resolves against the local MCP server like any other tool.
        _SYSTEM_TOOLS = {
            FILE_WRITE_TOOL, FILE_READ_TOOL, GREP_TOOL, BASH_TOOL, EXPLORE_TOOL,
        }
        if tool_name in _SYSTEM_TOOLS and step.server_id != SYSTEM_TOOLS_SERVER_ID:
            logger.info(
                f"[resolve] '{tool_name}' is a system tool — pinning to the "
                f"smartloop server (was '{step.server_name}')"
            )
            step.server_id = SYSTEM_TOOLS_SERVER_ID
            step.server_name = SYSTEM_TOOLS_SERVER_NAME

        is_system = (
            step.server_id == SYSTEM_TOOLS_SERVER_ID or tool_name in _SYSTEM_TOOLS
        )

        # Find server + tool.
        server = None
        if step.server_id:
            server = self.tool_caller.find_server(mcp_servers, step.server_id, step.server_name)

        if not is_system:
            # External MCP tools resolve through the MCPAgent: it retrieves the
            # top tools from the mcp_tools vector index as one flat
            # server.tool_name list and the LLM picks one. There is no
            # keyword-scan fallback. An explicit, already-resolvable plan choice
            # (server present + exact tool schema) is honoured first so a
            # well-specified plan skips retrieval.
            explicit_ok = (
                server is not None
                and has_explicit_tool
                and self.tool_caller.find_tool_schema(server, tool_name) is not None
            )
            if not explicit_ok:
                picked = self._mcp_agent.select_tool(
                    directive=directive,
                    project_dir=project_dir,
                    mcp_servers=mcp_servers,
                )
                if picked:
                    picked_server_id, picked_tool = picked
                    server = self.tool_caller.find_server(
                        mcp_servers, picked_server_id, picked_server_id,
                    )
                    if server:
                        tool_name = picked_tool
                        has_explicit_tool = True
                        logger.info(
                            f"[resolve] MCPAgent selected '{tool_name}' on "
                            f"server '{server.name}' via vector retrieval"
                        )

        if not server:
            logger.warning(
                f"[resolve] Server not found for tool '{tool_name}': "
                f"{step.server_name} (id={step.server_id})"
            )
            return None

        # Tool schema — exact lookup only (no keyword guessing).
        tool_schema = self.tool_caller.find_tool_schema(server, tool_name)
        if not tool_schema:
            logger.warning(
                f"[resolve] Tool '{tool_name}' not found on server "
                f"'{server.name}' — skipping step"
            )
            return None

        input_schema = tool_schema.get('input_schema', {}) or {}

        # When the step comes from an ```mcp block, scope extraction to
        # only the declared args.  inferred_args are extracted by the LLM;
        # default_args are filled in later.  This prevents the LLM from
        # hallucinating values for undeclared optional parameters.
        if step.inferred_args is not None:
            # Only expose inferred_args to the LLM for extraction.
            # default_args are fixed values — they are merged
            # unconditionally after extraction, so the LLM must not
            # see (and potentially override) them.
            allowed_keys = set(step.inferred_args)
            properties = input_schema.get("properties", {})
            scoped_props = {k: v for k, v in properties.items() if k in allowed_keys}
            scoped_required = [r for r in input_schema.get("required", []) if r in allowed_keys]
            input_schema = {
                **input_schema,
                "properties": scoped_props,
                "required": scoped_required,
            }
            logger.info(
                f"[resolve] Scoped '{tool_name}' schema to inferred args: "
                f"{sorted(allowed_keys)}"
            )
        else:
            # Prompt-derived step: only extract REQUIRED arguments. Optional
            # params are left unset so the tool applies its own defaults — this
            # stops the LLM fabricating optional filters (e.g. Linear
            # list_issues' priority/query/includeArchived) that wrongly narrow
            # or empty the result. Tools with no required params get no
            # inferred arguments at all.
            required = input_schema.get("required", [])
            properties = input_schema.get("properties", {})
            scoped_props = {k: v for k, v in properties.items() if k in required}
            input_schema = {
                **input_schema,
                "properties": scoped_props,
                "required": required,
            }
            logger.info(
                f"[resolve] Scoped '{tool_name}' schema to required args only: "
                f"{sorted(required)}"
            )

        server_url = self.tool_caller.get_server_url(server)

        if not server_url and server.server_type != MCPServerType.LOCAL:
            logger.warning(f"[resolve] No server URL configured for: {step.server_name}")
            return None

        # Dependency analysis
        all_tools = [
            {'name': t.name, 'description': t.description, 'input_schema': t.input_schema}
            for t in server.tools
        ]
        dep_context = directive
        if previous_results:
            prior_text = "\n".join(
                f"[{r.get('tool_name', 'tool')}]: {str(r.get('result', ''))[:500]}"
                for r in previous_results
            )
            dep_context = f"{directive}\n\nPrevious step results:\n{prior_text}"

        dep_analysis = self.dep_analyzer.invoke(
            user_input=dep_context,
            tool_name=tool_name,
            input_schema=input_schema,
            available_tools=all_tools,
        )
        if dep_analysis.has_dependencies and not dep_analysis.can_execute:
            logger.info(f"[resolve] '{tool_name}' has unresolvable dependencies; falling through")
            return None

        resolved_ids = {}
        if dep_analysis.has_dependencies and dep_analysis.can_execute:
            resolved_ids = self.tool_caller.execute_prerequisites(
                dep_analysis=dep_analysis,
                server=server,
                user_message=directive,
                input_schema=input_schema,
                previous_results=previous_results,
            )

        # Extract arguments — use the user's original message as the
        # primary input so the LLM extracts values from it, not from
        # skill guidance text that may contain metadata like skill names.
        #
        # For ```mcp block steps, use the user message plus the reasoning
        # report from the upstream reason node.  The report
        # contains the structured analysis that should inform tool
        # arguments (e.g. issue title/description derived from the
        # analysis).  Raw explore dumps and tool descriptions are
        # excluded — they introduce noise that small models pick up
        # as parameter values.
        if step.inferred_args is not None:
            # The reasoning report flows through prev_results from the
            # reason node (tool_name=REASON_NODE).
            reasoning_report = None
            if previous_results:
                for pr in previous_results:
                    if pr.get("tool_name") == REASON_NODE:
                        reasoning_report = str(pr.get("result", ""))
                        break
            if reasoning_report:
                extraction_input = (
                    f"{user_message}\n\n"
                    f"---\nAnalysis report:\n{reasoning_report[:4000]}\n---"
                )
            else:
                extraction_input = user_message
        else:
            # NOTE: Do NOT inject the tool description into the
            # extraction input — small models copy it verbatim into
            # same-named params (e.g. a `description` field). Values
            # come from the user message + upstream results only.
            #
            # When a reasoning report exists from upstream steps, use it
            # as the primary data source so the LLM derives parameter
            # values from actual findings — not from the raw user message.
            skill_report = ctx.skill_report
            if skill_report and previous_results:
                extraction_input = (
                    f"User request: {user_message}\n\n"
                    f"---\nAnalysis report (use these findings to fill parameter values):\n"
                    f"{self._response.strip_tool_code_blocks(skill_report[:4000])}\n---"
                )
            elif previous_results:
                prior_parts = [
                    str(r.get('result', ''))[:4000]
                    for r in previous_results
                ]
                prior_data = "\n".join(prior_parts)
                extraction_input = (
                    f"User request: {user_message}\n\n"
                    f"---\nData from previous step (use this to fill parameter values):\n"
                    f"{prior_data}\n---"
                )
            else:
                extraction_input = f"{user_message}"

        # NOTE: Do NOT append step.instruction to extraction_input.
        # The instruction may contain skill body text (descriptions,
        # tool names, etc.) that small models pick up as parameter
        # values.  The tool schema already tells the LLM what
        # parameters are needed; the user message + previous results
        # provide the values.

        # Try to fill all required params deterministically before invoking
        # the LLM.  When every required param is a URL and the URL is in the
        # user message, skip the LLM extraction entirely.
        required_params_pre = input_schema.get('required', [])
        properties_pre = input_schema.get('properties', {})
        default_keys = set(step.default_args or {})
        deterministic_args = {}
        all_deterministic = bool(required_params_pre)
        for pn in required_params_pre:
            if pn in default_keys:
                continue
            pi = properties_pre.get(pn, {})
            if self.arg_extractor._is_url_parameter(pn, pi):
                urls = self.arg_extractor._extract_urls_from_text(user_message)
                if not urls:
                    urls = self.arg_extractor._extract_urls_from_text(directive)
                if urls:
                    deterministic_args[pn] = urls[0]
                    continue
            all_deterministic = False
            break

        if not input_schema.get('properties'):
            # No required/declared params to extract — call the tool with no
            # inferred arguments and let it apply its own defaults.
            arguments = {}
            logger.info("[resolve] '%s' has no extractable params — no inferred arguments", tool_name)
        elif all_deterministic and deterministic_args:
            arguments = deterministic_args
            logger.info(
                "[resolve] '%s' all required params filled deterministically: %s",
                tool_name, arguments,
            )
        elif resolved_ids:
            arguments = self.arg_extractor.invoke_with_resolved_ids(
                user_input=extraction_input,
                tool_name=tool_name,
                input_schema=input_schema,
                resolved_ids=resolved_ids,
                llm=llm,
            )
        else:
            arguments = self.arg_extractor.invoke(
                user_input=extraction_input,
                tool_name=tool_name,
                input_schema=input_schema,
                previous_tool_results=previous_results,
                llm=llm,
            )

        # Merge skill-level default args.  When the step originates from
        # an ```mcp block, default_args are fixed values declared by the
        # skill author — they always win.  For other steps (skill_map /
        # LLM-derived) they act as fallbacks for missing values.
        if step.default_args:
            force = step.inferred_args is not None  # mcp block → fixed
            for dk, dv in step.default_args.items():
                if force or dk not in arguments or not arguments[dk]:
                    arguments[dk] = dv
                    logger.info(f"[resolve] '{tool_name}' applying default: {dk}={dv}")

        # Drop the extractor's bookkeeping key. The required-params check
        # below re-derives what is actually missing and can still satisfy
        # URL params deterministically from the directive — so we don't skip
        # on the extractor's report alone (e.g. a fetch whose URL is stated
        # in the instruction but which the LLM failed to extract).
        arguments.pop('_missing_params', None)

        required_params = input_schema.get('required', [])
        properties = input_schema.get('properties', {})

        # Deterministically fill a missing file-write `content` from the data
        # accumulated by upstream steps.  The content to write IS the prior
        # step output (fetched page, generated text, analysis report) — asking
        # a small model to regurgitate a large blob back into a JSON string is
        # unreliable and was leaving `content` empty, so the step got skipped.
        # Passing the accumulated data through directly is the whole point of
        # the fetch → generate → write chain.
        if (tool_name == FILE_WRITE_TOOL
                and 'content' in required_params
                and not arguments.get('content')
                and previous_results):
            accumulated = "\n\n".join(
                str(r.get('result', '')).strip()
                for r in previous_results
                if str(r.get('result', '')).strip()
            )
            if accumulated:
                arguments['content'] = accumulated
                logger.info(
                    f"[resolve] '{tool_name}' filled 'content' from "
                    f"{len(previous_results)} accumulated step result(s) "
                    f"({len(accumulated)} chars)"
                )

        still_missing = []
        for param_name in required_params:
            val = arguments.get(param_name)
            if val is None or val == "" or val == "null":
                param_info = properties.get(param_name, {})
                if self.arg_extractor._is_url_parameter(param_name, param_info):
                    urls = self.arg_extractor._extract_urls_from_text(directive)
                    if not urls:
                        urls = self.arg_extractor._extract_urls_from_text(user_message)
                    if urls:
                        arguments[param_name] = urls[0]
                    else:
                        still_missing.append(param_name)
                else:
                    still_missing.append(param_name)
        if still_missing:
            logger.info(f"[resolve] '{tool_name}' still missing required params ({', '.join(still_missing)}); skipping")
            return None

        invalid = self.tool_caller.validate_argument_values(arguments, properties)
        if invalid:
            hard_invalid = [msg for msg in invalid if any(
                f"'{p}'" in msg for p in required_params
            )]
            if hard_invalid:
                logger.info(f"[resolve] '{tool_name}' invalid required argument values: {hard_invalid}; falling through")
                return None
            for msg in invalid:
                for p in list(arguments):
                    if f"'{p}'" in msg and p not in required_params:
                        logger.info(f"[resolve] '{tool_name}' dropping invalid optional param '{p}': {msg}")
                        del arguments[p]

        # Remove params not in the tool schema.  default_args keys are
        # exempt — they were declared by the skill author and may not be
        # in the scoped schema used for LLM extraction.
        full_properties = (tool_schema.get('input_schema') or {}).get('properties', {})
        default_keys = set(step.default_args or {})
        if full_properties:
            unknown = [
                k for k in list(arguments)
                if k not in full_properties and k not in default_keys
            ]
            if unknown:
                logger.info(f"[resolve] '{tool_name}' removing unknown params: {unknown}")
                for k in unknown:
                    del arguments[k]

        # Strip placeholder values
        _PLACEHOLDER_RE = re.compile(
            r'not\s+(?:specified|provided|available|given)|'
            r'needs?\s+to\s+be\s+provided|'
            r'n/?a\b|unknown',
            re.IGNORECASE,
        )
        for k in list(arguments):
            val = arguments[k]
            if isinstance(val, str) and _PLACEHOLDER_RE.search(val):
                if k not in required_params:
                    logger.info(f"[resolve] '{tool_name}' dropping placeholder optional param '{k}': {val[:60]}")
                    del arguments[k]

        logger.info(f"[resolve] '{tool_name}' resolved arguments: {arguments}")
        return server, tool_name, arguments

    def call_tool(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: Dict[str, Any],
        cwd: Optional[str] = None,
    ) -> Optional[tuple]:
        """Call an MCP tool with pre-resolved arguments.

        ``cwd`` is forwarded to local (stdio) servers so built-in system
        tools resolve relative paths against the project work dir.

        Returns ``(formatted_content, raw_content)`` or ``None`` on failure.
        """
        access_token = self.tool_caller.get_access_token(server)
        server_url = self.tool_caller.get_server_url(server)

        logger.info(f"[execute] Calling '{tool_name}' with arguments: {arguments}")
        result = self.tool_caller.call_tool(
            server=server, tool_name=tool_name,
            arguments=arguments, access_token=access_token, cwd=cwd,
        )

        if not result.success and result.error and "401" in result.error and server_url:
            new_token = self.tool_caller.refresh_access_token(server)
            if new_token:
                result = self.tool_caller.call_tool(
                    server=server, tool_name=tool_name,
                    arguments=arguments, access_token=new_token, cwd=cwd,
                )

        if not result.success:
            logger.warning(f"[execute] '{tool_name}' failed: {result.error}")
            return None
        if not result.content or result.is_error:
            error_detail = ""
            if result.content:
                raw = " ".join(
                    block.get('text', '') if isinstance(block, dict) else str(block)
                    for block in result.content
                ).strip()
                if raw:
                    error_detail = f": {raw[:300]}"
            logger.warning(f"[execute] '{tool_name}' returned empty or error result{error_detail}")
            return None

        formatted_content = self.result_formatter.format_tool_result(
            tool_name=tool_name, content=result.content, is_error=result.is_error,
        )
        raw_parts = [
            block.get('text', '') if isinstance(block, dict) and block.get('type') == 'text'
            else str(block)
            for block in result.content
        ]
        raw_content = "\n".join(raw_parts)

        logger.info(
            f"[execute] '{tool_name}' returned {len(formatted_content)} chars "
            f"({self.orchestrator_llm.get_num_tokens(formatted_content)} tokens)"
        )
        logger.info(f"[execute] '{tool_name}' raw result: {raw_content[:1000]}")
        return formatted_content, raw_content

    def build_available_servers(self, mcp_servers: List[MCPServer]) -> List[Dict[str, Any]]:
        available = []
        for server in mcp_servers:
            if not server.enabled:
                continue
            if not server.tools:
                continue

            available.append({
                'id': server.id,
                'name': server.name,
                'description': server.description,
                'summary': server.summary,
                'server_type': server.server_type,
                'tools': [
                    {
                        'name': t.name,
                        'description': t.description,
                        'input_schema': t.input_schema,
                    }
                    for t in server.tools
                ],
            })

        return available