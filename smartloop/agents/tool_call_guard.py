"""Detect hallucinated tool calls in a model's user-facing answer.

Small instruct models trained with a tool-calling format (Gemma's
```` ```tool_code ````, Llama's bare ``{"name": ..., "parameters": ...}``)
will sometimes answer a *skill* prompt by emitting a call to a tool that
does not exist — e.g. a "recommend a workout" skill producing::

    ```tool_code
    {"tool_call": "recommend_workout_routine", "parameters": {"day_of_week": "Friday"}}
    ```

Nothing downstream can execute that: by the time the response model runs,
tool execution is already over and its results are in the prompt.  The block
is not an answer, so it must never reach the user — the caller retries with
:data:`NO_TOOL_CALL_DIRECTIVE` appended instead.
"""

import json
import re
from typing import Optional

#: Fence tags that are *only* ever used for tool invocation. The orchestrator
#: already strips these from context it builds (``_strip_tool_code_blocks``);
#: here they mark a whole response as an execution directive, not an answer.
EXEC_FENCE_TAGS = frozenset({"tool_code", "tool_call", "mcp", "tool"})

#: Keys that name the tool being invoked, in the formats small models emit.
_TOOL_KEYS = ("tool_call", "tool_name", "toolname", "function_call", "function")

#: Keys that carry the invocation's arguments.
_ARG_KEYS = ("parameters", "arguments", "args", "parameter", "input")

_FENCE_RE = re.compile(
    # A single fenced block spanning the whole text. The closing fence is
    # optional: a model that hits its token limit mid-call leaves it open.
    r"\A```(?P<tag>[^\n`]*)\n(?P<body>.*?)(?:```|\Z)\s*\Z",
    re.DOTALL,
)

NO_TOOL_CALL_DIRECTIVE = (
    "Answer the request directly, in prose. You have NO tools and cannot call "
    "any — every tool named above has already run and its output (if any) is "
    "included above. Never emit a tool call, a function call, or a JSON object "
    "describing one, and never wrap your answer in a ```tool_code, ```mcp or "
    "```json fence. Write the answer itself."
)


def _first_json_object(text: str) -> Optional[dict]:
    """Parse ``text`` as a JSON object, tolerating trailing prose.

    Returns ``None`` when it does not start with a parseable object.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(stripped)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _is_tool_call_payload(obj: dict) -> bool:
    """True when a JSON object describes a tool invocation.

    Both a tool name *and* an argument bag are required. One alone is too
    common in legitimate JSON answers — a "name" field appears in half the
    objects a model is ever asked to produce.
    """
    keys = {str(k).lower() for k in obj}
    named = any(k in keys for k in _TOOL_KEYS) or (
        # ``{"name": ..., "arguments": ...}`` — the OpenAI function shape.
        "name" in keys and any(k in keys for k in _ARG_KEYS)
    )
    return named and any(k in keys for k in _ARG_KEYS)


def is_tool_call_only(text: str) -> bool:
    """True when the entire answer is a tool call rather than a response.

    Covers a fenced ``tool_code``/``mcp`` block, a fenced JSON tool call, and
    a bare tool-call object with no prose around it.
    """
    if not text:
        return False
    stripped = text.strip()

    match = _FENCE_RE.match(stripped)
    if match:
        tag = match.group("tag").strip().lower()
        if tag in EXEC_FENCE_TAGS:
            return True
        body = match.group("body")
    else:
        body = stripped

    payload = _first_json_object(body)
    return payload is not None and _is_tool_call_payload(payload)


def may_be_tool_call_prefix(text: str) -> bool:
    """True when ``text`` could still be the opening of a tool-call answer.

    Used to hold a partial stream back: once the response is complete
    :func:`is_tool_call_only` decides for real.  Being wrong here costs only
    streaming smoothness, so it errs toward holding.
    """
    if not text:
        return False
    stripped = text.lstrip()
    if stripped.startswith("```"):
        head = stripped[3:]
        tag = head.split("\n", 1)[0].strip().lower()
        if tag in EXEC_FENCE_TAGS:
            return True
        # A ```json fence is a legitimate answer shape on its own; only the
        # presence of tool-call keys makes it a leaked call.
        lowered = head.lower()
        return any(f'"{k}"' in lowered for k in _TOOL_KEYS)
    if stripped.startswith("{"):
        lowered = stripped.lower()
        return any(f'"{k}"' in lowered for k in _TOOL_KEYS)
    return False
