"""MCPAgent — resolves an MCP step to a concrete tool via vector retrieval.

Selection is two-stage ("vector retrieves, LLM picks"), but over a *single
flat tool list* — there is no separate server-selection step:

1. Semantic retrieval over the per-project ``mcp_tools`` collection (populated
   when a server is registered) narrows all installed tools down to the top-K
   candidates for the step instruction.  Each candidate is one tool identified
   by its qualified ``server.tool_name``.
2. The LLM picks the single best entry from that one flat list.

The agent only performs *selection*; the orchestrator's execute handler still
calls the tool (via :class:`MCPToolCaller`) and returns the result to the LLM
for the step — mirroring how the ``FetchAgent`` feeds a step's result back.
"""

import logging
import re
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MCPAgent:
    """Vector-retrieval tool selector over a single flat ``server.tool_name`` list."""

    def __init__(self, llm: Any, logger: logging.Logger = logger):
        self._llm = llm
        self._logger = logger

    def select_tool(
        self,
        directive: str,
        project_dir: str,
        mcp_servers: List[Any],
        base_dir: Optional[str] = None,
        top_k: int = 5,
        llm: Optional[Any] = None,
    ) -> Optional[Tuple[str, str]]:
        """Return ``(server_id, tool_name)`` for *directive*, or None.

        Retrieves candidate tools from the ``mcp_tools`` vector index as one
        flat list (each entry a ``server.tool_name``) and asks the LLM to pick
        one. Returns None when nothing is indexed or the LLM declines.

        ``llm`` overrides the model used for the pick — the caller passes the
        step's capability model, which judges "which of these tools does the
        user mean" far better than the small orchestrator model.  Mirrors the
        override :class:`~smartloop.agents.mcp.explorer.Explorer` takes for the
        same reason.  Falls back to the agent model when not supplied.
        """
        if not project_dir or not directive:
            return None

        try:
            from smartloop.db.mcp_tool_indexer import MCPToolIndexer
            indexer = MCPToolIndexer(
                project_dir=project_dir, logger=self._logger, base_dir=base_dir,
            )
            candidates = indexer.search(directive, n_results=top_k)
        except Exception as e:
            self._logger.warning(f"[mcp-agent] tool retrieval failed: {e}")
            return None

        if not candidates:
            self._logger.info("[mcp-agent] no candidate tools retrieved for step")
            return None

        # Build ONE flat tool list, each entry keyed by its qualified
        # server.tool_name. No grouping by server, no separate server selector.
        entries: List[Tuple[str, str]] = []   # (server_id, tool_name)
        lines: List[str] = []
        seen: set = set()
        for c in candidates:
            md = c.metadata or {}
            server_id = md.get("server_id")
            tool_name = md.get("tool_name")
            qualified = md.get("qualified_name") or (
                f"{md.get('server_name', '')}.{tool_name}" if tool_name else None
            )
            if not server_id or not tool_name or qualified in seen:
                continue
            server = next((s for s in mcp_servers if getattr(s, "id", None) == server_id), None)
            if server is None or not getattr(server, "enabled", True):
                continue
            seen.add(qualified)
            desc = self._tool_description(server, tool_name) or (c.text or "")
            idx = len(entries) + 1
            entries.append((server_id, tool_name))
            label = f"[{idx}] {qualified}"
            if desc:
                label += f": {desc}"
            lines.append(label)

        if not entries:
            return None

        # A single candidate needs no disambiguation.
        if len(entries) == 1:
            self._logger.info("[mcp-agent] single candidate → %s", lines[0])
            return entries[0]

        prompt = (
            f"User request: {directive}\n\n"
            "Available tools (server.tool_name):\n"
            + "\n".join(lines)
            + "\n\nWhich tool number best fits the request? "
            "Reply with ONLY the number, or 0 if none apply."
        )

        self._logger.info(
            "[mcp-agent] %d candidate tool(s) → LLM pick over one flat list",
            len(entries),
        )

        try:
            response = (llm or self._llm).invoke(prompt)
        except Exception as e:
            self._logger.warning(f"[mcp-agent] LLM pick failed: {e}")
            return None

        choice = self._parse_choice(str(response) if response is not None else "")
        if choice is None or choice < 1 or choice > len(entries):
            self._logger.info("[mcp-agent] LLM chose no tool among candidates")
            return None

        server_id, tool_name = entries[choice - 1]
        self._logger.info("[mcp-agent] selected %s", lines[choice - 1])
        return server_id, tool_name

    @staticmethod
    def _parse_choice(text: str) -> Optional[int]:
        """Extract the first integer from the LLM reply, or None."""
        m = re.search(r"-?\d+", text)
        return int(m.group()) if m else None

    @staticmethod
    def _tool_description(server: Any, tool_name: str) -> str:
        """Return a tool's description from the live server object, if present."""
        for t in getattr(server, "tools", None) or []:
            name = getattr(t, "name", None) if not isinstance(t, dict) else t.get("name")
            if name == tool_name:
                if isinstance(t, dict):
                    return t.get("description", "") or ""
                return getattr(t, "description", "") or ""
        return ""
