"""Tool fan-out — calling one tool once per item.

When a plan step is marked ``for_each`` ("for each fund, get the ticker
info"), the list of items has to be recovered from upstream context before the
tool can run at all.  That extraction is the awkward part: upstream results are
free text or JSON from another tool, and a small model asked for a list will
happily return prose, a numbered list, or a JSON array.

All the calls for one step collapse into a single DAG node.  They share the
same tool and differ only by argument, so fanning them across nodes would buy
nothing and cost a node each.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from smartloop.agents.nodes import NodeResult
from smartloop.agents.mcp.explorer import SearchContext
from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
from smartloop.agents.orchestration.context import _OrchestratorContext
from smartloop.agents.orchestration.model_router import ModelRouter
from smartloop.agents.orchestration.response_builder import ResponseBuilder
from smartloop.agents.orchestration.step_executor import (
    StepExecutor, plan_step_result,
)
from smartloop.agents.planner import PlanStep
from smartloop.workspace.workspace import MCPServer

logger = logging.getLogger(__name__)


class ToolFanout:
    """Runs one tool once per item extracted from upstream context."""

    # Ceiling on items per fan-out.  The list comes from an LLM reading
    # upstream context, so a degenerate answer could otherwise queue an
    # unbounded number of tool calls.
    _FANOUT_MAX_ITEMS = 20

    # Ceiling on items per fan-out.  The list comes from an LLM reading
    # upstream context, so a degenerate answer could otherwise queue an
    # unbounded number of tool calls.
    def __init__(
        self,
        models: ModelRouter,
        steps: StepExecutor,
        response: ResponseBuilder,
        tool_caller: MCPToolCaller,
    ):
        self._models = models
        self._steps = steps
        self._response = response
        self.tool_caller = tool_caller

    def make_handler(
        self,
        ctx: _OrchestratorContext,
        step: PlanStep,
        search_ctx: SearchContext,
    ) -> Callable:
        """Create a handler that extracts items and calls a tool once per item.

        Three phases: extract item list, resolve+execute per item, aggregate.

        When all arguments are pre-filled (common case: single inferred arg
        like ``symbol``), the tool is called directly — no LLM-based
        dependency analysis or argument extraction per item.
        """
        def handler(deps: Dict) -> NodeResult:
            node_id = f"step_{step.step}_{step.tool_name}_fanout"
            prev_results = self._response.collect_raw_from_deps(deps)

            items = self._extract_fanout_items(ctx, step, prev_results, search_ctx)
            if not items:
                logger.warning(f"[fanout] No items extracted for {step.tool_name}")
                return plan_step_result(node_id, step, "")

            logger.info(f"[fanout] Extracted {len(items)} item(s) for {step.tool_name}")

            server = self._find_fanout_server(step, ctx.mcp_servers, ctx.available_servers)
            if not server:
                logger.warning(f"[fanout] Server not found for {step.tool_name}")
                return plan_step_result(node_id, step, "")

            tool_name = step.tool_name or ""

            all_results: List[Dict[str, Any]] = []
            for i, item_value in enumerate(items):
                arguments = dict(step.default_args or {})

                if isinstance(item_value, dict):
                    arguments.update(item_value)
                elif isinstance(item_value, str) and step.inferred_args and len(step.inferred_args) == 1:
                    arguments[step.inferred_args[0]] = item_value

                logger.info(f"[fanout] Calling {tool_name} [{i+1}/{len(items)}]: {arguments}")

                result = self._steps.call_tool(server, tool_name, arguments, cwd=ctx.effective_cwd)
                if result:
                    formatted, raw = result
                    all_results.append({
                        "item": item_value,
                        "tool_name": tool_name,
                        "content": formatted,
                        "raw": raw,
                    })

            if not all_results:
                return plan_step_result(node_id, step, "")

            combined_content = "\n\n---\n\n".join(
                f"[{r['tool_name']} for {r['item']}]:\n{r['content']}"
                for r in all_results
            )
            combined_raw = "\n\n---\n\n".join(
                f"[{r['tool_name']} for {r['item']}]:\n{r['raw']}"
                for r in all_results
            )

            logger.info(
                f"[fanout] {step.tool_name} completed {len(all_results)}/{len(items)} calls"
            )
            return plan_step_result(node_id, step, combined_content, combined_raw)

        return handler

    def _find_fanout_server(
        self,
        step: PlanStep,
        mcp_servers: List[MCPServer],
        available_servers: List[Dict[str, Any]],
    ) -> Optional[MCPServer]:
        """Locate the MCP server for a fan-out step."""
        tool_name = step.tool_name or ""
        if step.server_id:
            server = self.tool_caller.find_server(mcp_servers, step.server_id, step.server_name)
            if server:
                return server
        for srv in mcp_servers:
            if not srv.enabled or not srv.tools:
                continue
            for t in srv.tools:
                t_name = t.name if hasattr(t, "name") else t.get("name", "")
                if t_name == tool_name:
                    return srv
        return None

    def _extract_fanout_items(
        self,
        ctx: _OrchestratorContext,
        step: PlanStep,
        prev_results: List[Dict[str, Any]],
        search_ctx: SearchContext,
    ) -> List[Any]:
        """Extract iteration items from upstream context.

        Uses a structured LLM prompt that asks for one value per line
        (more reliable with small models than asking for JSON arrays).
        Previous step results are included so the LLM can see the
        already-extracted data.
        """
        context_parts: List[str] = []
        for r in prev_results:
            result_text = str(r.get("result", ""))[:4000]
            if result_text:
                context_parts.append(result_text)
        if search_ctx.vector_context:
            context_parts.append(search_ctx.vector_context[:6000])

        context = "\n\n".join(context_parts)
        if not context:
            return []

        arg_name = ""
        if step.inferred_args:
            arg_name = step.inferred_args[0]

        skill_hint = ""
        if ctx.skill_context:
            skill_hint = f"Skill instruction: {ctx.skill_context[:500]}\n\n"

        prompt = (
            "<system>\n"
            f"Your task: read the context and list every distinct '{arg_name or 'item'}' "
            f"value that should be passed to the tool '{step.tool_name}'.\n\n"
            "Output format: one value per line, nothing else.\n"
            "Do not add numbering, bullets, or explanation.\n"
            "Do not skip any values — list ALL of them.\n"
            "</system>\n"
            "<user>\n"
            f"{skill_hint}"
            f"Context:\n{context[:8000]}\n\n"
            f"List every '{arg_name or 'item'}' value, one per line:\n"
            "</user>"
        )

        try:
            response = str(self._models.agent_llm.invoke(prompt)).strip()
            logger.info(f"[fanout-extract] LLM response: {response[:300]}")

            items = self._parse_fanout_response(response)
            if items:
                if len(items) > self._FANOUT_MAX_ITEMS:
                    logger.warning(
                        f"[fanout-extract] Truncated {len(items)} items "
                        f"to {self._FANOUT_MAX_ITEMS}"
                    )
                return items[:self._FANOUT_MAX_ITEMS]
        except Exception as e:
            logger.warning(f"[fanout-extract] extraction failed: {e}")

        return []

    @staticmethod
    def _parse_fanout_response(response: str) -> List[str]:
        """Parse fan-out extraction response: one value per line or JSON array."""
        import json as _json
        text = response.strip()
        if not text:
            return []

        # Try JSON array first (in case the LLM returns one)
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                items = _json.loads(json_match.group())
                if isinstance(items, list) and items:
                    return [
                        str(v) if not isinstance(v, str) else v
                        for v in items if v
                    ]
            except (ValueError, TypeError):
                pass

        # Parse one-value-per-line format
        items: List[str] = []
        for line in text.splitlines():
            val = line.strip().strip("-•*").strip()
            val = re.sub(r'^\d+[\.\)]\s*', '', val).strip()
            if not val or len(val) > 200:
                continue
            if val.lower().startswith(("here ", "the ", "these ", "below", "value")):
                continue
            items.append(val)

        return items