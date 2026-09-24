"""One read-only MCP tool call, for a background pass that makes one.

The account probe (:mod:`smartloop.mcp.account`) needs a tool call that changes
nothing, its text back, and a failure treated as a skip rather than an
exception — including the expired-OAuth-token retry the live tool path does.
The 401 retry in particular is the kind of thing that gets forgotten in a
re-implementation and then only shows up as a background pass that silently
stops working an hour into a session, so it lives in one place.

The transport underneath is :class:`~smartloop.agents.mcp.mcp_tool_caller.MCPToolCaller`
— the same class the live agent path calls tools through, so a background call
and a model-issued one reach a server identically.  Only its transport-level
helpers are used here; the argument extractor and result mapper it takes for
the live path are LLM-backed and have no place in a background pass.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

# Characters kept from a failed call's reason.  Long enough to keep the part of
# a server's validation message that names the offending field — truncating
# before it turns the one log line that explains a failure into a line that
# doesn't.
MAX_REASON_CHARS = 500


class ToolCallResult(NamedTuple):
    """Outcome of one read-only call.

    A NamedTuple so existing ``payload, changed, error = ...`` unpacking keeps
    working while the fields get names at the call sites that want them.
    """

    payload: Optional[str]
    #: True when an expired token was refreshed, so the caller can persist it.
    credentials_changed: bool
    error: Optional[str]


def content_to_text(content: Optional[Sequence[dict]]) -> str:
    """Join an MCP tool result's content blocks into one string."""
    if not content:
        return ""
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
            else:
                parts.append(json.dumps(block, default=str))
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p).strip()


def failure_reason(result: Any) -> str:
    """Short reason a tool call produced nothing.

    MCP servers report an argument rejection as a *successful* call with
    ``is_error`` set and the message in the content, so the content is read too
    — otherwise the most common failure ("query is required") shows up as a
    bare "no content".
    """
    if getattr(result, "error", None):
        return str(result.error)[:MAX_REASON_CHARS]
    content = getattr(result, "content", None)
    if content:
        text = content_to_text(content)
        if text:
            return text[:MAX_REASON_CHARS]
    return "empty result"


def ensure_tool_caller(existing: Any, logger: logging.Logger) -> Optional[Any]:
    """Return *existing*, or build the default transport caller lazily."""
    if existing is not None:
        return existing
    try:
        from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
        return MCPToolCaller(arg_extractor=None, result_mapper=None)
    except Exception as e:
        logger.warning(f"[mcp] No MCP tool caller available: {e}")
        return None


def call_read_only(
    caller: Any,
    server: Any,
    tool_name: str,
    arguments: Dict[str, Any],
    logger: logging.Logger,
    label: str = "mcp",
) -> ToolCallResult:
    """Call one read-only tool and return its text.

    A failure is a skip, not an exception: a background pass covers what it can
    reach and the next one tries again — but the reason travels back so a pass
    that produced nothing can say why.  An expired OAuth token is refreshed once
    and the call retried, mirroring the live tool path.
    """
    if caller is None:
        return ToolCallResult(None, False, "no MCP tool caller available")
    try:
        result, credentials_changed = caller.call_with_refresh(
            server=server, tool_name=tool_name, arguments=arguments,
        )
        if not result.success or result.is_error or not result.content:
            reason = failure_reason(result)
            logger.info(
                "[%s] %s.%s(%s) returned nothing: %s",
                label, getattr(server, "name", "?"), tool_name,
                ", ".join(arguments) or "-", reason,
            )
            return ToolCallResult(None, credentials_changed, reason)
        return ToolCallResult(
            content_to_text(result.content) or None, credentials_changed, None,
        )
    except Exception as e:
        logger.info(
            "[%s] %s.%s raised: %s",
            label, getattr(server, "name", "?"), tool_name, e,
        )
        return ToolCallResult(None, False, str(e)[:MAX_REASON_CHARS])
