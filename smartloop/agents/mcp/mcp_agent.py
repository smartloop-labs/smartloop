"""MCPAgent — resolves an MCP step to a concrete tool call, arguments included.

Registering a server enumerates its tools once (see
``ModelService._provision_mcp_server``) and persists what it found: the name,
the description and the JSON input schema of every tool.  That list *is* the
catalog, and it is already in memory on every request — so resolution reads it
directly.  The tools are shown to the model as callable signatures and it
answers with the one call to make, its required arguments filled from the
request and from what earlier steps returned.

This replaces a vector index over tool descriptions.  Embedding a one-line
description and taking the top-K by cosine distance answered "which tool reads
like this sentence", which is not the question being asked: a step names an
*operation*, and the tool that serves it is the one whose arguments the context
can actually fill.  Retrieval also had to run before argument extraction, so
the two halves of a single decision were made by two calls that could not see
each other — the tool was picked without knowing whether its arguments could be
satisfied, and a wrong pick was then argued for confidently.  One call over the
whole catalog costs no embedding pass at registration, no per-project vector
collection, and no re-index when a server's tools change.

A model that can reason *always* reasons here.  The call passes
``thinking=True`` and an output budget above the floor under which
``LlamaCpp._thinking_for_budget`` suppresses the block, so neither the user's
thinking setting nor a small default budget can turn it off.  The setting is
about the visible answer; this call's output is a JSON tool call whose
monologue is stripped before anything reads it
(:mod:`smartloop.llms.thinking`), and choosing a tool while working out whether
its arguments can be filled is precisely the judgement the block exists for.
The flag travels with the request, so it holds whether the weights are in this
process or behind the model host.

Which model runs it is decided one level up: ``pipeline_builder`` routes the
resolve node to a model declaring ``ModelCapability.THINKING`` when the
workspace has one enabled, rather than to whichever model the step would
otherwise use.
"""

import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.utils.json import parse_json_markdown

from smartloop.agents.planner import FETCH_SERVER_ID, SYSTEM_TOOLS_SERVER_ID

logger = logging.getLogger(__name__)

# Output budget for the resolve call when the model reasons.  Above
# ``LlamaCpp.THINKING_MIN_BUDGET`` (512) so the reasoning block is not
# suppressed: the JSON answer is ~60 tokens and the rest is room to think.
RESOLVE_MAX_TOKENS = 1024

# Catalog trimming.  The whole tool list goes into one prompt now, so it is
# bounded here rather than by retrieval: a long description is guidance the
# model needs the gist of, not all of.
_MAX_TOOL_DESC = 200
_MAX_PARAM_DESC = 140
_MAX_CATALOG_CHARS = 12000
# Never render less than this, whatever the window says — below it the list is
# too abbreviated to pick from at all.
_MIN_CATALOG_CHARS = 4000
_MAX_CONTEXT_CHARS = 2000

# How much of a resolved call is shown to the user (see `format_call`).
_MAX_SHOWN_ARGS = 4
_MAX_SHOWN_ARG_CHARS = 60

# One question resolves the same way twice in a single request: once on the
# collect path (Explorer runs a read-only tool while gathering context) and
# again when the plan reaches its TOOL step.  The second call would ask the
# same model the same question over the same catalog — seconds of reasoning for
# an answer already in hand — so the answer is kept.  Keyed by everything that
# could change it, tiny, and process-wide because the two callers hold
# different MCPAgent instances.
_CACHE_SIZE = 4
_cache_lock = threading.Lock()
_resolution_cache: "OrderedDict[tuple, Optional[ResolvedCall]]" = OrderedDict()


def clear_resolution_cache() -> None:
    """Forget every remembered resolution (tests, and a workspace reload)."""
    with _cache_lock:
        _resolution_cache.clear()


def _cache_key(directive: str, context: Optional[str], entries) -> tuple:
    return (
        " ".join(directive.split()).lower(),
        hash((context or "").strip()),
        tuple(e.qualified for e in entries),
    )


@dataclass(frozen=True)
class ResolvedCall:
    """One tool call: where it lives, what it is, and what to pass it.

    ``arguments`` holds the tool's *required* parameters, as far as the request
    and the context could fill them — the caller decides whether a partial fill
    is worth calling with (see ``StepExecutor._resolve_tool_arguments``).
    """

    server_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    reasoned: bool = False


def format_call(server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    """``notion.create-pages(parent="Roadmap", pages=[…])`` — the call, for a person.

    What was decided is the interesting part of an MCP step, and until this
    existed the client only saw which *model* had been chosen: a tool ran, or
    silently did not, with the call itself visible in the server logs alone.
    Values are truncated because an argument can be a whole document, and
    rendered with ``repr`` so an empty string or a stray newline is visible
    instead of vanishing into the punctuation.
    """
    qualified = f"{server_name}.{tool_name}" if server_name else tool_name
    if not arguments:
        return f"{qualified}()"
    parts = []
    for key, value in list(arguments.items())[:_MAX_SHOWN_ARGS]:
        shown = value if isinstance(value, str) else repr(value)
        if len(shown) > _MAX_SHOWN_ARG_CHARS:
            shown = shown[:_MAX_SHOWN_ARG_CHARS] + "…"
        parts.append(f"{key}={shown!r}" if isinstance(value, str) else f"{key}={shown}")
    if len(arguments) > _MAX_SHOWN_ARGS:
        parts.append(f"+{len(arguments) - _MAX_SHOWN_ARGS} more")
    return f"{qualified}({', '.join(parts)})"


def missing_required(input_schema: Dict[str, Any], arguments: Dict[str, Any]) -> List[str]:
    """Required parameters of *input_schema* that *arguments* leaves unfilled."""
    missing = []
    for name in input_schema.get("required", []) or []:
        value = arguments.get(name)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(name)
    return missing


class MCPAgent:
    """Resolves a step to a tool call by reasoning over the discovered tools."""

    def __init__(self, llm: Any, logger: logging.Logger = logger):
        self._llm = llm
        self._logger = logger
        self._validators = None

    # ------------------------------------------------------------------ resolve

    def resolve(
        self,
        directive: str,
        mcp_servers: List[Any],
        context: Optional[str] = None,
        llm: Optional[Any] = None,
    ) -> Optional[ResolvedCall]:
        """Return the tool call *directive* asks for, or None.

        *context* is what earlier steps produced, in whatever shape the caller
        already had it — the values a chained call needs ("close the issue I
        just filed") live there and nowhere else, so resolution reads it
        alongside the request.

        ``llm`` overrides the model.  Callers pass the step's capability model:
        this is the harder of the judgements an MCP step makes, and leaving it
        on the small router model while upgrading only the arguments got the
        arguments right for the wrong tool.
        """
        if not directive:
            return None

        entries = self._catalog(mcp_servers or [])
        if not entries:
            self._logger.info("[mcp-agent] no discovered tools to resolve against")
            return None

        # A lone tool that takes no required arguments is already resolved;
        # there is nothing for the model to decide or fill.
        if len(entries) == 1 and not entries[0].schema.get("required"):
            only = entries[0]
            self._logger.info(
                "[mcp-agent] single argument-free tool → %s", only.qualified,
            )
            return ResolvedCall(only.server_id, only.tool_name)

        key = _cache_key(directive, context, entries)
        with _cache_lock:
            if key in _resolution_cache:
                cached = _resolution_cache[key]
                _resolution_cache.move_to_end(key)
                self._logger.info(
                    "[mcp-agent] reusing this question's resolution: %s",
                    format_call(
                        "", cached.tool_name, cached.arguments,
                    ) if cached else "no tool",
                )
                return cached

        effective_llm = llm or self._llm
        prompt = self._prompt(directive, context, entries, effective_llm)
        reasoned = self.can_think(effective_llm)
        budget = self._budget(effective_llm, prompt, RESOLVE_MAX_TOKENS)

        self._logger.info(
            "[mcp-agent] resolving over %d discovered tool(s) with %s "
            "(thinking=%s, max_tokens=%s)",
            len(entries), self._model_label(effective_llm), reasoned, budget,
        )

        try:
            # thinking=True regardless of what `reasoned` says: a model that
            # cannot reason ignores it, and one that can must not have this
            # call decided by the user's answer-formatting setting.
            response = effective_llm.invoke(
                prompt, max_tokens=budget, thinking=True,
            )
        except TypeError:
            # An LLM handle from before the per-call flag (or a test double
            # with a narrower signature) — the pick is still worth making.
            response = effective_llm.invoke(prompt, max_tokens=budget)
        except Exception as e:
            self._logger.warning(f"[mcp-agent] tool resolution failed: {e}")
            return None

        text = response if isinstance(response, str) else str(response or "")
        choice, raw_arguments = self._parse(text, entries)
        if choice is None:
            self._logger.info(
                "[mcp-agent] no tool resolved from: %s", text.strip()[:200],
            )
            self._remember(key, None)
            return None

        entry = entries[choice]
        arguments = self._clean_arguments(entry.schema, raw_arguments)
        self._logger.info(
            "[mcp-agent] resolved %s",
            format_call(entry.server_name, entry.tool_name, arguments),
        )
        resolved = ResolvedCall(
            server_id=entry.server_id,
            tool_name=entry.tool_name,
            arguments=arguments,
            reasoned=reasoned,
        )
        self._remember(key, resolved)
        return resolved

    @staticmethod
    def _remember(key: tuple, resolved: Optional[ResolvedCall]) -> None:
        """Keep this resolution for the other caller in the same request."""
        with _cache_lock:
            _resolution_cache[key] = resolved
            _resolution_cache.move_to_end(key)
            while len(_resolution_cache) > _CACHE_SIZE:
                _resolution_cache.popitem(last=False)

    # ------------------------------------------------------------------ catalog

    @dataclass(frozen=True)
    class _Entry:
        server_id: str
        server_name: str
        tool_name: str
        qualified: str
        description: str
        schema: Dict[str, Any]

    def _catalog(self, mcp_servers: List[Any]) -> List["MCPAgent._Entry"]:
        """Every callable external tool, as discovery recorded it.

        The system server is left out: its tools (bash, grep, file-read/write,
        explore) are resolved by the planner against their own constants, and
        putting them in this list only gives the model a way to answer "read a
        file" to a question about Notion.
        """
        entries: List[MCPAgent._Entry] = []
        seen: set = set()
        for server in mcp_servers:
            server_id = getattr(server, "id", None)
            if not server_id or server_id in (SYSTEM_TOOLS_SERVER_ID, FETCH_SERVER_ID):
                continue
            if getattr(server, "system", False):
                continue
            if not getattr(server, "enabled", True):
                continue
            server_name = getattr(server, "name", None) or server_id
            for tool in getattr(server, "tools", None) or []:
                name = self._field(tool, "name")
                if not name:
                    continue
                qualified = f"{server_name}.{name}"
                if qualified in seen:
                    continue
                seen.add(qualified)
                entries.append(MCPAgent._Entry(
                    server_id=server_id,
                    server_name=server_name,
                    tool_name=name,
                    qualified=qualified,
                    description=(self._field(tool, "description") or "").strip(),
                    schema=self._field(tool, "input_schema") or {},
                ))
        return entries

    @staticmethod
    def _field(tool: Any, name: str) -> Any:
        """Read *name* off a tool, which is a model on one path and a dict on another."""
        if isinstance(tool, dict):
            return tool.get(name)
        return getattr(tool, name, None)

    def _signature(
        self, entry: "MCPAgent._Entry", number: int, compact: bool = False,
    ) -> str:
        """One catalog entry: the call, its purpose, and the arguments it needs.

        Only *required* parameters are listed.  An optional one the model can
        see is one it can fabricate — a priority filter or a stale cursor that
        silently narrows a listing to nothing — and the tool applies its own
        defaults for everything left unset.

        ``compact`` drops the prose and keeps the call shape, for a catalog too
        long to render in full (see :meth:`_prompt`).
        """
        schema = entry.schema or {}
        properties = schema.get("properties") or {}
        required = [r for r in (schema.get("required") or []) if r in properties]

        if compact:
            args = ", ".join(
                f"{name}: {(properties.get(name) or {}).get('type', 'any')}"
                for name in required
            )
            return f"[{number}] {entry.qualified}({args})"

        head = f"[{number}] {entry.qualified}"
        if entry.description:
            head += f" — {self._trim(entry.description, _MAX_TOOL_DESC)}"
        if not required:
            return f"{head}\n      (no required arguments)"

        lines = [head]
        for name in required:
            info = properties.get(name) or {}
            line = f"      {name} ({info.get('type', 'any')})"
            desc = (info.get("description") or "").strip()
            if desc:
                line += f": {self._trim(desc, _MAX_PARAM_DESC)}"
            enum = info.get("enum")
            if enum:
                line += f" [one of: {', '.join(str(v) for v in enum[:8])}]"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _trim(text: str, limit: int) -> str:
        flat = re.sub(r"\s+", " ", text).strip()
        return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"

    def _catalog_budget(self, llm: Any) -> int:
        """How many characters of catalog this model can be shown.

        Measured against the model actually running the call rather than a
        fixed number: a 13k-context model has room for 42 tools *with* their
        descriptions, and the descriptions are what the pick is made on — a
        list of bare names had a 7.5B answer "none of these apply" to a
        question its Notion server plainly served.  A third of the window
        leaves the rest for the request, the prior results and the reasoning
        block.
        """
        try:
            n_ctx = llm.get_max_tokens()
            if isinstance(n_ctx, int) and n_ctx > 0:
                # ~3.5 chars per token holds for tool names and prose alike; a
                # rough conversion is enough for a budget.
                return max(_MIN_CATALOG_CHARS, int(n_ctx / 3 * 3.5))
        except Exception:
            pass
        return _MAX_CATALOG_CHARS

    def _prompt(
        self,
        directive: str,
        context: Optional[str],
        entries: List["MCPAgent._Entry"],
        llm: Any = None,
    ) -> str:
        # Every tool has to be in the list — a tool the model cannot see is a
        # tool it cannot pick, and dropping the tail of a 42-tool catalog means
        # the answer depends on the order the servers happened to be
        # registered in.  So the *descriptions* go first when the catalog is
        # too long, not the tools: a bare `server.tool(arg: type)` line is a
        # fifth the size and still identifies the call.  Only if even that
        # overflows are entries dropped, and then the log says how many.
        budget = self._catalog_budget(llm) if llm is not None else _MAX_CATALOG_CHARS
        signatures = [self._signature(e, i) for i, e in enumerate(entries, start=1)]
        if sum(len(b) for b in signatures) > budget:
            self._logger.info(
                "[mcp-agent] %d tool(s) exceed %d chars — listing call shapes "
                "without descriptions", len(entries), budget,
            )
            signatures = [
                self._signature(e, i, compact=True)
                for i, e in enumerate(entries, start=1)
            ]

        used, kept = 0, []
        for block in signatures:
            used += len(block)
            if used > budget and kept:
                self._logger.warning(
                    "[mcp-agent] catalog truncated to %d of %d tool(s) at %d "
                    "chars — the rest cannot be picked for this step",
                    len(kept), len(entries), budget,
                )
                break
            kept.append(block)
        signatures = kept

        context_block = ""
        if context and context.strip():
            context_block = (
                "Context from earlier steps (use it to fill arguments):\n"
                f"{self._trim(context, _MAX_CONTEXT_CHARS)}\n\n"
            )

        return f"""\
<system>
You resolve a request into exactly ONE tool call, chosen from the numbered list.
Respond with ONLY a JSON object: {{"tool": <number>, "arguments": {{...}}}}

Rules:
- "tool" MUST be one of the numbers in the list, or 0. Never null, never a name,
  never a number that is not listed. No other keys, no commentary, no fences.
- The request names what the user wants, not the tool. A request about content
  a listed server holds ("the pitch doc in Notion", "my open issues") is served
  by that server's search or read tool — pick it and put the subject of the
  request in its query argument. Reading is cheap and a near miss is useful;
  answer 0 only when no listed server could hold what is being asked about.
- "arguments" holds that tool's listed arguments, taken from the request or the
  context. The text after an argument name is a DESCRIPTION of the field —
  guidance, never a value.
- Omit an argument whose value is not stated in the request or the context.
  Never invent an id, a cursor, a token or a URL.

Example: for "outline the roadmap page in Notion" with a listed
[7] Notion.notion-search(query: string), answer {{"tool": 7, "arguments":
{{"query": "roadmap"}}}}.
</system>
<user>
Request: {directive}

{context_block}Tools:
{chr(10).join(signatures)}

JSON:
</user>"""

    # ------------------------------------------------------------------ parsing

    def _parse(
        self, text: str, entries: List["MCPAgent._Entry"],
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """``(index into entries, raw arguments)``, or ``(None, {})``.

        Accepts what the models actually emit: the JSON object asked for, the
        same object with the tool named instead of numbered, a bare number from
        a model that ignored the shape, and — last — a reply that declined
        (``"tool": null``) while naming a tool somewhere in its own prose.  An
        explicit ``0`` is never salvaged: that is the model saying no, and it
        is allowed to.
        """
        stripped = re.sub(r"```\w*\s*", "", text).strip().rstrip("`")
        if not stripped:
            return None, {}

        payload: Any = None
        try:
            payload = parse_json_markdown(stripped)
        except Exception:
            payload = None

        if isinstance(payload, list) and payload:
            payload = payload[0]

        if isinstance(payload, dict):
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            picked = payload.get("tool", payload.get("name", payload.get("tool_name")))
            index = self._resolve_choice(picked, entries)
            if index is not None:
                return index, arguments
            if picked == 0:
                return None, {}
            # Declined without saying 0 — a small model answers `"tool": null`
            # with a note about what the tools are.  If that note names one,
            # take it: re-asking costs another reasoning pass.
            index = self._named_in(stripped, entries)
            return (index, arguments) if index is not None else (None, {})

        # A bare number ("2", "Tool 2") — no arguments to be had.
        index = self._resolve_choice(stripped, entries)
        if index is not None:
            return index, {}
        index = self._named_in(stripped, entries)
        return (index, {}) if index is not None else (None, {})

    @staticmethod
    def _named_in(
        text: str, entries: List["MCPAgent._Entry"],
    ) -> Optional[int]:
        """The first catalog tool named anywhere in *text*, or None.

        Longest name first, so ``notion-search`` is not shadowed by a shorter
        name it contains.
        """
        low = text.lower()
        best: Optional[Tuple[int, int]] = None   # (position, index)
        for i, entry in enumerate(entries):
            for name in (entry.qualified.lower(), entry.tool_name.lower()):
                at = low.find(name)
                if at != -1 and (best is None or at < best[0]):
                    best = (at, i)
        return None if best is None else best[1]

    @staticmethod
    def _resolve_choice(
        picked: Any, entries: List["MCPAgent._Entry"],
    ) -> Optional[int]:
        """Turn a number or a tool name into an index into *entries*."""
        if isinstance(picked, bool) or picked is None:
            return None
        if isinstance(picked, (int, float)):
            number = int(picked)
            return number - 1 if 1 <= number <= len(entries) else None

        if not isinstance(picked, str):
            return None
        name = picked.strip()
        if not name:
            return None

        for i, entry in enumerate(entries):
            if name.lower() in (entry.qualified.lower(), entry.tool_name.lower()):
                return i

        # "2", "[2]", "Tool 2: linear.create_issue" — the leading number wins,
        # but only when the reply is not mostly prose about a named tool.
        match = re.search(r"\d+", name)
        if match:
            number = int(match.group())
            if 1 <= number <= len(entries):
                return number - 1
        return None

    # ---------------------------------------------------------------- arguments

    def _clean_arguments(
        self, schema: Dict[str, Any], raw: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep the required arguments that carry a real value.

        The same filters the argument extractor applies, for the same reasons:
        a small model answers "MISSING", echoes the field's description back as
        its value, or fabricates a cursor it cannot know.  Anything the tool
        does not declare is dropped rather than sent.
        """
        if not isinstance(raw, dict) or not raw:
            return {}
        properties = (schema or {}).get("properties") or {}
        required = set((schema or {}).get("required") or [])
        validators = self._argument_validators()

        cleaned: Dict[str, Any] = {}
        for key, value in raw.items():
            if key not in properties or (required and key not in required):
                continue
            if value is None or value in ("", "null", "MISSING", [], {}):
                continue
            info = properties.get(key) or {}
            if validators is not None:
                if validators._is_cursor_parameter(key):
                    continue
                if validators._is_schema_echo(value, info):
                    continue
                if validators._is_placeholder_echo(key, value, info):
                    continue
            cleaned[key] = value
        return cleaned

    def _argument_validators(self):
        """The extractor, used for its value checks only (never for a call).

        Reusing it keeps one definition of "this is a fabricated id / an echoed
        description" instead of a second copy that drifts from the first.
        """
        if self._validators is None:
            try:
                from .argument_extractor import MCPArgumentExtractor
                self._validators = MCPArgumentExtractor(self._llm)
            except Exception as e:  # pragma: no cover - import-time only
                self._logger.debug("[mcp-agent] argument validators unavailable: %s", e)
                return None
        return self._validators

    # -------------------------------------------------------------------- model

    @staticmethod
    def can_think(llm: Any) -> bool:
        """True when *llm* is a model that can reason.

        Both LLM handles answer this the same way — ``LlamaCpp`` from the flag
        bound onto its ``Llama`` at load (bound only for a template that reads
        ``enable_thinking``, including a user-built or merged model the model
        registry says nothing about), and ``RemoteLLM`` by asking the host,
        which owns the weights.  Reported in the log line and used to route the
        resolve node; the call itself asks for reasoning either way.
        """
        return bool(getattr(llm, "supports_thinking", False))

    @staticmethod
    def _model_label(llm: Any) -> str:
        """A short name for the model in a log line, never an exception."""
        label = getattr(llm, "model_label", None)
        if isinstance(label, str) and label:
            return label
        try:
            from smartloop.constants import display_model_label
            meta = llm.model.metadata or {}
            name = display_model_label(meta.get("general.name", ""))
            if name:
                return name
        except Exception:
            pass
        return type(llm).__name__

    @staticmethod
    def _budget(llm: Any, prompt: str, want: int) -> int:
        """*want* tokens, or as many as fit after the prompt.

        ``LlamaCpp._call`` takes a requested ``max_tokens`` at face value, and
        ``prompt + output`` over ``n_ctx`` is rejected outright by llama.cpp —
        so a whole-catalog prompt has to ask for what is actually left.
        """
        try:
            headroom = llm.get_max_tokens() - llm.get_num_tokens(prompt) - 64
            if isinstance(headroom, int):
                return max(64, min(want, headroom))
        except Exception:
            pass
        return want
