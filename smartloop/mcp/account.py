"""Which account a connection is linked to.

A connected card in the UI used to say "Connected", which restated what the
presence of tools, an unlink button and a re-index button already proved.  What
it could not answer is the question someone actually has in front of two Notion
connections: *whose* Notion is this?

MCP has no standard "who am I", so this resolves one of the identity tools
servers conventionally ship (``notion-get-self``, Linear's ``get_viewer``, …),
calls it through the same read-only path the content crawl uses, and pulls a
human-readable label out of whatever shape comes back.

Everything here is best-effort: a server with no identity tool, or one that
refuses the call, simply has no account label and the card shows what it showed
before.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from smartloop.mcp.crawl_plans import normalize_name
from smartloop.mcp.read_only_calls import call_read_only, ensure_tool_caller

logger = logging.getLogger(__name__)

# Tools that answer "who is this connection authenticated as".
#
# Matched *exactly* (see :func:`resolve_account_tool`), unlike crawl steps.  The
# loose matcher would pair ``get_me`` with Trello's ``get_members`` — a member
# list is not an account, and labelling a connection with the wrong person's
# name is worse than labelling it with nothing.  So every real spelling is
# listed instead.
ACCOUNT_TOOLS: Tuple[str, ...] = (
    "notion-get-self", "notion_get_self", "API-get-self", "get_self",
    "get_viewer", "viewer", "get_current_user", "current_user",
    "get_authenticated_user", "get_me", "users_me", "whoami", "who_am_i",
    "get_my_profile", "get_account",
)

# Keys whose value names the account, in the order they are preferred.
#
# Email first: it is the one label that distinguishes two connections to the
# same product for certain.  ``workspace_name`` next, because a Notion
# integration token authenticates as a bot whose ``name`` is the integration
# ("Smartloop") — the workspace is the identity a person recognises there.
ACCOUNT_KEYS: Tuple[str, ...] = (
    "email", "workspace_name", "workspaceName",
    "displayName", "display_name", "username", "login", "name",
)

# How deep to look for an account key.  Identity payloads nest the useful part
# (Notion puts the human behind ``bot.owner.user.person.email``), but an
# unbounded walk over an arbitrary result would happily label a connection with
# the first ``name`` in an unrelated list.
_MAX_DEPTH = 6

# A label longer than this is not a name — it is a description, a token or a
# whole record that happened to sit under a matching key.
_MAX_LABEL_CHARS = 80


def resolve_account_tool(tool_names: List[str]) -> Optional[str]:
    """Return the server's identity tool, or None.

    Exact-match only, on the shared name normalisation:
    :func:`~smartloop.mcp.crawl_plans.resolve_tool` widens to prefix and
    substring passes, which is right for a crawl step (skipping one is a missed
    page) and wrong here (guessing wrong names the connection after someone
    else).
    """
    by_norm = {normalize_name(name): name for name in tool_names if name}
    for candidate in ACCOUNT_TOOLS:
        match = by_norm.get(normalize_name(candidate))
        if match:
            return match
    return None


def _label_from(value: Any) -> Optional[str]:
    if not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip()
    if not text or len(text) > _MAX_LABEL_CHARS:
        return None
    return text


def extract_account(payload: Optional[str]) -> Optional[str]:
    """Pull an account label out of an identity tool's result.

    Walks the parsed payload breadth-first so the shallowest match wins, and
    tries every key in :data:`ACCOUNT_KEYS` at each depth before going deeper —
    an email three levels down beats a display name at the top, but only
    because email is the better label, not because it is deeper.
    """
    if not payload:
        return None

    text = payload.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # A server that answers in prose: the whole line is the best label
        # available, when it is short enough to be one.
        return _label_from(text.splitlines()[0] if text else None)

    # Collect every candidate with the depth it was found at, then rank by key
    # preference first and depth second.
    found: Dict[str, Tuple[int, str]] = {}

    def walk(node: Any, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                label = _label_from(value)
                if label and key in ACCOUNT_KEYS and (
                    key not in found or depth < found[key][0]
                ):
                    found[key] = (depth, label)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value, depth + 1)
        elif isinstance(node, list):
            # Only a single-element list is unambiguous: "the" account.  A list
            # of users is a directory, and picking one of them is a guess.
            if len(node) == 1:
                walk(node[0], depth + 1)

    walk(data, 0)
    for key in ACCOUNT_KEYS:
        if key in found:
            return found[key][1]
    return None


def probe_account(server: Any, tool_caller: Any = None) -> Optional[str]:
    """Return the account *server* is linked to, or None.

    One read-only call, made through the same path as the content crawl.  Never
    raises: an unreachable server or a missing identity tool leaves the
    connection unlabelled.
    """
    tool_names = [
        (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
        for t in (getattr(server, "tools", None) or [])
    ]
    tool_names = [n for n in tool_names if n]
    tool = resolve_account_tool(tool_names)
    if not tool:
        return None

    payload, _changed, error = call_read_only(
        ensure_tool_caller(tool_caller, logger), server, tool, {},
        logger, label="mcp-account",
    )
    if error or not payload:
        return None

    account = extract_account(payload)
    if account:
        logger.info(
            "[mcp-account] '%s' is linked as %s (via %s)",
            getattr(server, "name", "?"), account, tool,
        )
    return account
