"""
MCP Server Selector Agent - Selects the appropriate MCP tool based on user request.

Builds a unified context (user input + skills + available tools) and lets the LLM
decide whether the request maps to a tool call and, if so, which tool to use.
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from smartloop.llms import BaseLLM

logger = logging.getLogger(__name__)

# Tool name constants for built-in tools (registered via System Tools server).
BASH_TOOL = "bash"
GREP_TOOL = "grep"
FILE_READ_TOOL = "file-read"
FILE_WRITE_TOOL = "file-write"
EXPLORE_TOOL = "explore"


class ToolSelection(BaseModel):
    """Pydantic model for MCP tool selection."""
    order: int = Field(description="Execution order of the tool")
    server_id: str = Field(default="", description="Unique identifier of the MCP server")
    server_name: str = Field(description="Name of the MCP server")
    tool_name: str = Field(description="Name of the tool to execute")
    tool_index: int = Field(default=-1, description="Index of the tool in the server's tool list")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")


class ToolSelectionResult(BaseModel):
    """Pydantic model for MCP tool selection result."""
    selections: List[ToolSelection] = Field(description="List of selected tools with details")


class MCPServerSelector:
    """
    Agent that analyzes user requests and selects the most appropriate MCP
    server and tool.

    The selection flow builds up context in layers:
      user input  →  skills / instructions  →  available MCP tools
    and sends a single prompt to the LLM which decides whether the request
    qualifies as a tool call and, if so, which tool to invoke.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_tool_catalogue(
        self,
        available_servers: List[Dict[str, Any]],
        skills: Optional[List[str]] = None,
    ) -> tuple:
        """Build numbered tool catalogue from available servers.

        Returns:
            (tools_list, tool_lookup, server_summaries) where tools_list is
            a list of formatted label strings, tool_lookup maps index to
            server/tool info, and server_summaries is a list of summary strings.
        """
        tools_list: List[str] = []
        tool_lookup: Dict[int, Dict[str, Any]] = {}
        server_summaries: List[str] = []
        idx = 0

        for server in available_servers:
            server_id = server.get("id", "")
            server_name = server.get("name", "Unknown")
            server_desc = server.get("description", "")
            server_summary = server.get("summary", "")
            if server_summary:
                server_summaries.append(f"- {server_name}: {server_summary}")
            for ti, tool in enumerate(server.get("tools", [])):
                idx += 1
                t_name = tool.get("name", "unknown")
                t_desc = tool.get("description", "")
                tool_lookup[idx] = {
                    "server_id": server_id,
                    "server_name": server_name,
                    "tool_name": t_name,
                    "tool_index": ti,
                }
                label = f"[{idx}] {server_name} — {t_name}"
                if t_desc:
                    label += f": {t_desc}"
                elif server_desc:
                    label += f" ({server_desc})"
                tools_list.append(label)

        return tools_list, tool_lookup, server_summaries

    def invoke(
        self,
        user_message: str,
        available_servers: List[Dict[str, Any]],
        skills: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> ToolSelectionResult:
        """Analyze the user's message and select the most appropriate MCP tool.

        Args:
            user_message: The user's request/question.
            available_servers: List of server dicts with ``name``, ``tools`` keys.
            skills: Optional skill content strings that provide routing hints
                    or define special actions (e.g. bash).

        Returns:
            ToolSelectionResult with zero or one selections.
        """
        try:
            if not available_servers and not skills:
                logger.info("No MCP tools or skills available for selection")
                return ToolSelectionResult(selections=[])

            # --- Build numbered tool catalogue ---
            tools_list, tool_lookup, server_summaries = self._build_tool_catalogue(
                available_servers, skills
            )

            if not tools_list:
                return ToolSelectionResult(selections=[])

            tools_text = "\n".join(tools_list)

            # --- Build skills section ---
            skills_section = ""
            if skills:
                skills_block = "\n".join(f"- {s}" for s in skills)
                skills_section = (
                    "Instructions / Skills (use these to decide which tool fits best, "
                    "or whether the request requires a tool at all):\n"
                    f"{skills_block}\n\n"
                )

            # --- Build server summaries section ---
            summaries_section = ""
            if server_summaries:
                summaries_section = (
                    "Server summaries:\n"
                    + "\n".join(server_summaries)
                    + "\n\n"
                )

            # --- Build cwd section ---
            cwd_section = ""
            if cwd:
                cwd_section = (
                    f"Working directory: {cwd}\n"
                    "IMPORTANT: When the user refers to local files, directories, "
                    "README, codebase, or project structure, use the smartloop "
                    "tools (file-read, grep, explore) — NOT fetch or web tools. "
                    "Only use fetch/web tools for URLs (http/https).\n\n"
                )

            # --- Single LLM call with full context ---
            prompt = (
                f"User request: {user_message}\n\n"
                f"{cwd_section}"
                f"{skills_section}"
                f"{summaries_section}"
                f"Available tools:\n{tools_text}\n\n"
                "Based on the user request and the instructions above, which tool "
                "number should be used? If no tool is needed, reply with 0.\n\n"
                "Reply with ONLY the tool number."
            )

            prompt_tokens = self.llm.get_num_tokens(prompt)
            logger.info(
                "Tool selection prompt (%d tools, %d skills, %d tokens):\n%s",
                len(tools_list), len(skills or []), prompt_tokens, prompt,
            )

            response = self.llm.invoke(prompt)
            response_text = str(response).strip() if response else ""
            response_tokens = self.llm.get_num_tokens(response_text)

            logger.info("Tool selection LLM response (%d tokens): %s", response_tokens, response_text)

            # Parse the response — try extracting a tool number
            selected = self._parse_tool_number(response_text, tool_lookup)

            if not selected:
                logger.info("LLM selected no tool from response: %s", response_text)
                return ToolSelectionResult(selections=[])

            selection = ToolSelection(
                order=1,
                server_id=selected["server_id"],
                server_name=selected["server_name"],
                tool_name=selected["tool_name"],
                tool_index=selected["tool_index"],
                confidence=0.85,
            )
            logger.info(
                f"Selected tool: {selection.server_name}/{selection.tool_name} "
                f"(confidence={selection.confidence})"
            )
            return ToolSelectionResult(selections=[selection])

        except Exception as e:
            logger.error(f"Failed to select tools: {e}")
            return ToolSelectionResult(selections=[])

    def invoke_followup(
        self,
        user_message: str,
        available_servers: List[Dict[str, Any]],
        accumulated_results: List[Dict[str, str]],
        skills: Optional[List[str]] = None,
    ) -> ToolSelectionResult:
        """Decide whether a follow-up tool call is needed based on accumulated results.

        Args:
            user_message: The user's original request.
            available_servers: List of server dicts with ``name``, ``tools`` keys.
            accumulated_results: List of dicts with ``tool_name`` and
                ``formatted_result`` from previously executed tools.
            skills: Optional skill content strings.

        Returns:
            ToolSelectionResult with zero or one selections.
        """
        try:
            if not available_servers:
                return ToolSelectionResult(selections=[])

            tools_list, tool_lookup, server_summaries = self._build_tool_catalogue(
                available_servers, skills
            )

            if not tools_list:
                return ToolSelectionResult(selections=[])

            tools_text = "\n".join(tools_list)

            # Build accumulated results section (truncate each to ~2000 chars)
            results_parts = []
            already_called = set()
            for r in accumulated_results:
                t_name = r.get("tool_name", "unknown")
                already_called.add(t_name)
                result_text = r.get("formatted_result", "")
                if len(result_text) > 2000:
                    result_text = result_text[:2000] + "\n... (truncated)"
                results_parts.append(f"Tool: {t_name}\nResult:\n{result_text}")

            results_section = "\n\n".join(results_parts)
            already_called_text = ", ".join(already_called)

            # Include skills so they guide follow-up tool decisions
            skills_section = ""
            if skills:
                skills_block = "\n".join(f"- {s}" for s in skills)
                skills_section = (
                    "Instructions / Skills (use these to decide what to do next):\n"
                    f"{skills_block}\n\n"
                )

            prompt = (
                f"User's original request: {user_message}\n\n"
                f"{skills_section}"
                f"Tools already executed and their results:\n{results_section}\n\n"
                f"Available tools:\n{tools_text}\n\n"
                "Based on the user's request, the instructions above, and the tool "
                "results, does the request need another tool call to be fully answered?\n"
                "- If the instructions or results indicate a follow-up tool would add "
                "meaningful value (e.g., fetching content for URLs found in results, "
                "looking up details for items discovered, expanding or enriching the "
                "data), reply with the tool number.\n"
                "- If the results already fully answer the request, or no available "
                "tool would help, reply with 0.\n"
                f"- Do NOT re-call tools that were already used: {already_called_text}\n\n"
                "Reply with ONLY the tool number."
            )

            prompt_tokens = self.llm.get_num_tokens(prompt)
            logger.info(
                "Follow-up selection prompt (%d tools, %d prior results, %d tokens)",
                len(tools_list), len(accumulated_results), prompt_tokens,
            )

            response = self.llm.invoke(prompt)
            response_text = str(response).strip() if response else ""
            logger.info("Follow-up selection LLM response: %s", response_text)

            selected = self._parse_tool_number(response_text, tool_lookup)

            if not selected:
                logger.info("No follow-up tool needed")
                return ToolSelectionResult(selections=[])

            # Don't re-call the same tool
            if selected["tool_name"] in already_called:
                logger.info(
                    "Follow-up selected already-called tool '%s', skipping",
                    selected["tool_name"],
                )
                return ToolSelectionResult(selections=[])

            selection = ToolSelection(
                order=1,
                server_id=selected["server_id"],
                server_name=selected["server_name"],
                tool_name=selected["tool_name"],
                tool_index=selected["tool_index"],
                confidence=0.80,
            )
            logger.info(
                f"Follow-up tool selected: {selection.server_name}/{selection.tool_name}"
            )
            return ToolSelectionResult(selections=[selection])

        except Exception as e:
            logger.error(f"Failed to select follow-up tool: {e}")
            return ToolSelectionResult(selections=[])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_number(
        text: str,
        tool_lookup: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extract a tool number from the LLM response.

        Handles multiple response formats:
        - Plain number: ``3``
        - JSON: ``[{"tool_number": 3}]`` or ``{"tool_number": 3}``
        - Prose with number: ``The best tool is [3]`` / ``Tool number: 3``
        - Zero or no number means no match.
        """
        text = text.strip()
        if not text:
            return None

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try JSON first (handles [{"tool_number": N}] and bare numbers)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, int):
                if parsed > 0:
                    return tool_lookup.get(parsed)
                return None
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "tool_number" in item:
                        num = int(item["tool_number"])
                        if num > 0:
                            return tool_lookup.get(num)
                return None
            if isinstance(parsed, dict) and "tool_number" in parsed:
                num = int(parsed["tool_number"])
                if num > 0:
                    return tool_lookup.get(num)
                return None
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: extract first integer from the response text
        numbers = re.findall(r"\b(\d+)\b", text)
        for n_str in numbers:
            num = int(n_str)
            if num in tool_lookup:
                return tool_lookup[num]

        return None
