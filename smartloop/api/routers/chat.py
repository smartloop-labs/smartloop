"""Chat completion router with OpenAI-compatible endpoints"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
import threading
import uuid

from datetime import date
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from smartloop.api.model import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatStatusChunk,
    ChatMessage,
    Citation,
    Choice,
    ChunkChoice,
    DeltaContent,
    Usage,
)
from smartloop.api.services import ModelService, get_model_service, get_api_client
from smartloop.constants import SYSTEM_PROMPT, resolve_current_date
from smartloop.types import ModelCapability


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])

def _status_event(step: str, status: str, message: str) -> str:
    """Format a status event as an SSE data line.

    Serializes a :class:`ChatStatusChunk` (``object: "chat.status"``) so
    clients can distinguish these from OpenAI-compatible
    ``chat.completion.chunk`` content lines and track them against a schema.
    """
    payload = ChatStatusChunk(step=step, status=status, message=message).model_dump_json()
    return f"data: {payload}\n\n"

# Thread pool for running sync LLM operations
_executor = ThreadPoolExecutor(max_workers=1)


# A leading ``/<skill-name>`` in the prompt is an explicit request to fulfil the
# turn with that skill (e.g. ``/fund-info summarise Q3``). Capture the token and
# the remainder so the token can be matched against the project's skills.
_SLASH_SKILL_RE = re.compile(r"^\s*/([A-Za-z0-9][\w-]*)\s*(.*)$", re.DOTALL)


def _resolve_requested_skill(message: str, skills: list) -> tuple:
    """Resolve a leading ``/<skill-name>`` slash-command against *skills*.

    Returns ``(skill_name, cleaned_message)`` when the prompt opens with a
    slash-command that names one of the project's skills — matched case- and
    separator-insensitively, so ``/fund-info`` hits a ``fund_info`` skill. The
    remainder of the prompt (after the token) becomes the message; when the user
    typed only the token, a minimal ``Run the '<name>' skill.`` stands in.

    Returns ``(None, message)`` unchanged when there is no leading slash-command
    or it names no known skill, so an ordinary prompt that merely starts with a
    path (``/etc/hosts …``) is left untouched.
    """
    if not message:
        return None, message
    m = _SLASH_SKILL_RE.match(message)
    if not m:
        return None, message
    token, rest = m.group(1), m.group(2)
    norm = lambda s: re.sub(r"[\s_\-]+", "", (s or "").lower())
    wanted = norm(token)
    for s in skills:
        if norm(getattr(s, "name", "")) == wanted:
            cleaned = rest.strip() or f"Run the '{s.name}' skill."
            return s.name, cleaned
    return None, message


_TAG_ESCAPE = {
    "<system>": "&lt;system&gt;",
    "</system>": "&lt;/system&gt;",
    "<user>": "&lt;user&gt;",
    "</user>": "&lt;/user&gt;",
    "<assistant>": "&lt;assistant&gt;",
    "</assistant>": "&lt;/assistant&gt;",
}


def _extract_tool_content(orchestrator_context: str) -> str:
    """Extract the raw tool payload from the orchestrator's wrapped context."""
    if not orchestrator_context:
        return ""
    marker = "TOOL RESULTS:\n"
    start = orchestrator_context.find(marker)
    if start < 0:
        return orchestrator_context.strip()
    return orchestrator_context[start + len(marker):].strip()


def _escape_role_tags(text: str) -> str:
    """Escape literal role tags in *text* so they don't break the regex in
    ``create_messages_from_prompt``.  Fetched web content can contain
    ``<user>`` or ``</user>`` (e.g. HTML/chat tutorials); leaving them
    unescaped truncates the extracted prompt catastrophically and makes
    the model emit no output.
    """
    if not text:
        return text
    for tag, esc in _TAG_ESCAPE.items():
        text = text.replace(tag, esc)
    return text


def format_messages_to_prompt(
    messages: List[ChatMessage],
    context: str | None = None,
    cwd: str | None = None,
    project_info: str | None = None,
    current_date: str | None = None,
) -> str:
    """Convert OpenAI-style messages to the prompt format expected by the LLM.

    When *context* is provided (pre-formatted by the orchestrator), it is
    injected into the first user message as-is.  When no orchestrator context
    is available the message is sent without auxiliary context so the model
    answers from its own knowledge rather than mistaking directory listings
    or project metadata for a restrictive knowledge base.
    """
    prompt_parts = []

    system_content = SYSTEM_PROMPT
    for msg in messages:
        if msg.role == "system" and msg.content:
            system_content += "\n" + msg.content
    prompt_parts.append(f"<system>{system_content}</system>")

    # The current date is runtime grounding the model was never trained on, so
    # it rides with the current user turn (next to the question) rather than
    # buried in the system prompt where a small model is unlikely to attend to it.
    date_str = current_date or date.today().strftime("%B %d, %Y")

    context_injected = False
    last_user_idx = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"),
        None,
    )

    for i, msg in enumerate(messages):
        if msg.role == "system":
            continue
        elif msg.role == "assistant":
            prompt_parts.append(
                f"<assistant>{_escape_role_tags(msg.content or '')}</assistant>"
            )
        elif msg.role == "user":
            user_parts = []

            # Put the user's question FIRST so the model always sees a clear
            # question, then the tool/document context.  Otherwise a short
            # question (e.g. just a URL) gets buried after a large context
            # block and small models emit no response.
            if msg.content:
                user_parts.append(_escape_role_tags(msg.content))

            if not context_injected and i == last_user_idx:
                user_parts.append(f"[Current date: {date_str}]")
                if context:
                    user_parts.append(_escape_role_tags(context))
                context_injected = True

            user_content = "\n\n".join(user_parts)
            prompt_parts.append(f"<user>{user_content}</user>")

    return "\n".join(prompt_parts)


def _request_current_date(request) -> str:
    """Resolve the current date from the client's ``additional_message_params``.

    Uses the browser's timezone + clock when supplied so date-aware answers
    match the user's calendar; falls back to the server date otherwise.
    """
    params = getattr(request, "additional_message_params", None)
    return resolve_current_date(
        getattr(params, "user_date", None) if params else None,
        getattr(params, "tz", None) if params else None,
    )


def _resolve_message_attachments(
    messages: List[ChatMessage],
    model_service: ModelService,
) -> List[ChatMessage]:
    """Return a new message list where attachment asset IDs are resolved to markdown.

    For each user message containing ``attachments``, the markdown content of
    each referenced asset is prepended to the message text so the model can
    see the full document context inline.
    """
    resolved = []
    for msg in messages:
        if msg.role == "user" and msg.attachments:
            attachment_md = model_service.resolve_attachment_context(msg.attachments)
            if attachment_md:
                combined = "\n\n".join(filter(None, [attachment_md, msg.content]))
                resolved.append(ChatMessage(role=msg.role, content=combined))
            else:
                resolved.append(msg)
        else:
            resolved.append(msg)
    return resolved


def _compact_messages(
    messages: List[ChatMessage],
    model_service: ModelService,
    reserved_tokens: int = 2048,
    context_tokens: int = 0,
    max_context: int | None = None,
) -> List[ChatMessage]:
    """Trim oldest conversation turns when the message list exceeds the context window.

    Always keeps the first user message (original context) and the last user
    message (the current question).  Removes the oldest assistant+user pairs
    working forward from index 1 until the token count fits within
    ``max_context - reserved_tokens - context_tokens``.

    ``context_tokens`` accounts for document_context, project_skills, and
    mcp_context that will be injected into the final prompt.

    ``max_context`` is the window of the model that will actually synthesize
    the response (which may be smaller than the default model's window — see
    ``ModelService.get_response_max_context_tokens``).  Defaults to the
    default model's window for compatibility with existing callers.
    """
    if max_context is None:
        max_context = model_service.get_max_context_tokens()
    if max_context <= 0:
        return messages

    budget = max_context - reserved_tokens - context_tokens
    if budget <= 0:
        return messages

    # Quick check — if we're already within budget, return as-is
    full_text = "\n".join(m.content or "" for m in messages)
    used_tokens = model_service.count_tokens(full_text)

    if used_tokens <= budget:
        logger.info(
            f"Context window: {used_tokens}/{max_context} tokens used "
            f"({budget} budget, {reserved_tokens} reserved) — no compaction needed"
        )
        return messages

    original_count = len(messages)

    # Build a mutable copy
    compacted = list(messages)

    # Remove oldest turns in pairs (assistant+user or user+assistant) starting
    # from index 1 so that roles keep alternating.  Index 0 (first user message
    # with context) and the last message are always kept.
    while len(compacted) > 2:
        # Pop two messages at index 1 to remove a full turn pair
        compacted.pop(1)
        if len(compacted) > 2:
            compacted.pop(1)

        text = "\n".join(m.content or "" for m in compacted)
        if model_service.count_tokens(text) <= budget:
            break

    # Safety: ensure roles alternate after compaction.  If two consecutive
    # messages share the same role, merge them so the model never sees an
    # invalid sequence.
    merged: List[ChatMessage] = [compacted[0]]
    for msg in compacted[1:]:
        if msg.role == merged[-1].role:
            merged[-1] = ChatMessage(
                role=msg.role,
                content="\n".join(filter(None, [merged[-1].content, msg.content])),
            )
        else:
            merged.append(msg)
    compacted = merged

    compacted_text = "\n".join(m.content or "" for m in compacted)
    compacted_tokens = model_service.count_tokens(compacted_text)
    removed = original_count - len(compacted)
    logger.info(
        f"Context window: {used_tokens}/{max_context} tokens exceeded budget ({budget}), "
        f"compacted {removed} messages — now {compacted_tokens}/{max_context} tokens"
    )

    return compacted


def _build_messages_with_history(
    request: ChatCompletionRequest,
    history: List[dict],
) -> List[ChatMessage]:
    """Prepend already-loaded conversation *history* to the request messages.

    ``history`` is the session's recent turns (recency-based, alternation
    enforced) loaded once by the caller via
    ``ModelService.get_relevant_history`` and shared with the orchestrator,
    so it is not re-queried here.  Prepending lets the LLM reference earlier
    context (e.g. "create a table from that").
    """
    if not history:
        return request.messages

    history_messages = [
        ChatMessage(role=h["role"], content=h["content"])
        for h in history
    ]
    return history_messages + request.messages


async def stream_chunks(
    request: ChatCompletionRequest,
    model_service: ModelService,
) -> AsyncIterator[str]:
    """
    Convert sync LLM streaming to async SSE.

    Uses a queue and thread pool to bridge sync iterator to async generator.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Switch project if specified
    if request.project_id:
        try:
            model_service.set_current_project(request.project_id)
        except ValueError:
            logger.warning(f"Project not found: {request.project_id}, using current project")

    # Load the session's recent history ONCE and share it between the prompt
    # (prepended prior turns) and the orchestrator (search/planning context).
    _req_last_user = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), None
    )
    conversation_history = model_service.get_relevant_history(
        request.session_id, _req_last_user or "",
    ) if request.session_id and _req_last_user else []

    messages = _build_messages_with_history(request, conversation_history)

    # Resolve attachments (with status events if attachments exist)
    has_attachments = any(m.attachments for m in messages if m.role == "user")
    if has_attachments:
        yield _status_event("attachments", "processing", "Resolving attachments...")
    messages = _resolve_message_attachments(messages, model_service)
    if has_attachments:
        yield _status_event("attachments", "completed", "Attachments resolved")

    # Image attachments on the current turn are handed to the orchestrator so a
    # vision-capable model reads the pixels + the user's own question, instead
    # of a pre-processed text transcription that loses the visual content.
    _cur_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
    image_attachments = (
        model_service.get_asset_images(_cur_user.attachments)
        if _cur_user and _cur_user.attachments else []
    )

    # Get project skills and document context separately
    last_user_msg = next(
        (m.content for m in reversed(messages) if m.role == "user"), None
    )

    project_skills = [s for s in model_service.get_project_skills() if not s.system]

    # A leading ``/<skill-name>`` in the prompt is an explicit request to fulfil
    # this turn with that skill (e.g. ``/morning what is my agenda``). Resolve it
    # against the project's skills, strip the token from the prompt (the planning
    # copy and the final-answer message), and pass the skill name to the
    # orchestrator, which validates it against the request intent before use.
    requested_skill = None
    _cur_msg = next((m for m in reversed(messages) if m.role == "user"), None)
    if _cur_msg is not None and _cur_msg.content:
        requested_skill, cleaned = _resolve_requested_skill(
            _cur_msg.content, project_skills,
        )
        if requested_skill:
            _cur_msg.content = cleaned
            last_user_msg = cleaned
            logger.info("[skill] user-selected skill: %r", requested_skill)

    # Shared cancellation signal for the whole request. Created before
    # orchestration so a client disconnect during the planning phase (skill
    # matching, routing) interrupts those LLM calls too — not just the final
    # response stream. The generator's finally sets it on exit/disconnect.
    cancelled = threading.Event()

    # Orchestrator handles vector search, MCP tools, and explore in one step
    orchestrator_context = None
    citations = None
    skill_instruction = None
    is_mutation = False
    if last_user_msg:
        yield _status_event("tools", "processing", "Processing...")

        # Bridge orchestrator step activity into the SSE stream. The
        # orchestrator runs in a worker thread and reports each step via
        # progress_callback; hand those events back to the event loop
        # thread-safely and flush them as chat.status while it works, so the
        # client shows live progress instead of a frozen spinner.
        status_queue: asyncio.Queue = asyncio.Queue()

        def _progress(step: str, status: str, message: str) -> None:
            loop.call_soon_threadsafe(
                status_queue.put_nowait, _status_event(step, status, message)
            )

        orchestrate_future = loop.run_in_executor(
            None,
            lambda: model_service.orchestrate(
                last_user_msg, skills=project_skills, cwd=request.cwd,
                conversation_history=conversation_history,
                has_attachments=has_attachments,
                images=image_attachments,
                progress_callback=_progress,
                cancelled=cancelled,
                requested_skill=requested_skill,
                current_date=_request_current_date(request),
            ),
        )

        from smartloop.config.app_settings import AppSettings
        _orch_timeout = AppSettings().orchestrate_timeout_s
        _deadline = loop.time() + _orch_timeout if _orch_timeout > 0 else None

        while not orchestrate_future.done():
            if _deadline is not None and loop.time() > _deadline:
                # A native llama.cpp call can't be preempted mid-token, so we
                # signal cancellation (checked between chunks) and stop waiting
                # rather than hang the client. We don't await the future — a
                # truly stuck call would block forever; the executor thread
                # unwinds on its own once the current chunk returns.
                cancelled.set()
                logger.error(
                    "[orchestrate] exceeded %.0fs budget — cancelling and "
                    "returning a timeout to the client", _orch_timeout,
                )
                # Drain any progress events already queued before giving up.
                while not status_queue.empty():
                    yield status_queue.get_nowait()
                yield _status_event(
                    "tools", "error",
                    f"Orchestration timed out after {_orch_timeout:.0f}s",
                )
                timeout_chunk = ChatCompletionChunk(
                    model=request.model,
                    choices=[ChunkChoice(
                        index=0,
                        delta=DeltaContent(
                            role="assistant",
                            content=(
                                "**Request timed out.**\n\n"
                                "The assistant took too long analyzing and "
                                "running the matched skills/tools and was "
                                "stopped. This often means the loaded model is "
                                "too small to synthesize the gathered context. "
                                "Try a simpler request, or enable a larger "
                                "model in **Settings → Models**."
                            ),
                        ),
                        finish_reason="stop",
                    )],
                )
                yield f"data: {timeout_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                model_service.unload_auxiliary_models()
                return
            try:
                event = await asyncio.wait_for(status_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            yield event

        orchestrator_context, _, tool_error, citations, skill_instruction, is_mutation = (
            await orchestrate_future
        )
        # Flush any step events scheduled just before completion.
        while not status_queue.empty():
            yield status_queue.get_nowait()

        if tool_error:
            yield _status_event("tools", "error", tool_error)
        else:
            yield _status_event("tools", "completed", "Tools completed")

    # For mutation operations (save, create, delete, etc.) return the
    # success message directly without running it through the LLM.
    if is_mutation and orchestrator_context:
        message_id = request.message_id or f"msg-{uuid.uuid4().hex[:8]}"
        yield _status_event("streaming", "processing", "Generating response...")
        initial_chunk = ChatCompletionChunk(
            model=request.model,
            choices=[ChunkChoice(
                index=0,
                delta=DeltaContent(role="assistant"),
                finish_reason=None,
            )]
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"
        yield f"data: {json.dumps({'object': 'chat.message_id', 'message_id': message_id})}\n\n"
        # Stream synthesized content in small chunks for progressive rendering
        _STREAM_CHUNK_SIZE = 12
        for i in range(0, len(orchestrator_context), _STREAM_CHUNK_SIZE):
            content_chunk = ChatCompletionChunk(
                model=request.model,
                choices=[ChunkChoice(
                    index=0,
                    delta=DeltaContent(content=orchestrator_context[i:i + _STREAM_CHUNK_SIZE]),
                    finish_reason=None,
                )]
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0.01)
        done_chunk = ChatCompletionChunk(
            model=request.model,
            choices=[ChunkChoice(
                index=0,
                delta=DeltaContent(),
                finish_reason="stop",
            )]
        )
        yield f"data: {done_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        if request.session_id:
            if last_user_msg:
                # For image turns, fold the vision model's reading of the image
                # into the stored user turn so later turns (which no longer carry
                # the attachment) can still reference what the image contained.
                user_record = last_user_msg
                if image_attachments:
                    user_record = (
                        f"{last_user_msg}\n\n[Attached image — description]:\n"
                        f"{orchestrator_context}"
                    )
                model_service.append_to_session(request.session_id, "user", user_record)
            model_service.append_to_session(request.session_id, "assistant", orchestrator_context, message_id=message_id)

        _project_id = model_service.get_project_id()
        _prompt_tokens = model_service.count_tokens(last_user_msg or "")
        _completion_tokens = model_service.count_tokens(orchestrator_context)
        get_api_client().push_token_usage(
            project_id=_project_id,
            model=request.model,
            prompt_tokens=_prompt_tokens,
            completion_tokens=_completion_tokens,
        )
        return

    project_id = model_service.get_project_id()
    project_info = model_service.get_project_info()

    yield _status_event("preparing", "processing", "Preparing response...")

    # The final synthesis runs on the capability/large/default model, whose
    # context window can be *smaller* than the default router model's.  Size
    # the context against the model that will actually consume it — otherwise
    # web/document context overflows the response model, generation aborts,
    # the request falls back to the default model, and the search content
    # never reaches the LLM.
    response_capability = ModelCapability.GENERAL if orchestrator_context else None
    max_ctx = model_service.get_response_max_context_tokens(response_capability)
    if orchestrator_context and max_ctx > 0:
        # Size the context against the response model's real window, reserving
        # room for the fixed prompt overhead (system prompt, chat history,
        # template tags) PLUS a healthy output budget.  A flat reserve here
        # left almost no output tokens once the system prompt was added, so
        # grounded responses (e.g. F1 team standings) were cut off mid-sentence
        # at the model's max_tokens limit.
        output_reserve = 512
        overhead_prompt = format_messages_to_prompt(
            messages, context="", cwd=request.cwd, project_info=project_info,
            current_date=_request_current_date(request),
        )
        overhead_tokens = model_service.count_tokens(overhead_prompt)
        budget = max(256, max_ctx - overhead_tokens - output_reserve)
        ctx_tokens = model_service.count_tokens(orchestrator_context)
        if ctx_tokens > budget:
            chars_per_token = max(1, len(orchestrator_context) // max(1, ctx_tokens))
            orchestrator_context = orchestrator_context[: budget * chars_per_token]
            logger.info(
                "[context] Truncated orchestrator context from %d to %d tokens "
                "(overhead %d, output reserve %d) to fit response model window (%d)",
                ctx_tokens, budget, overhead_tokens, output_reserve, max_ctx,
            )

    messages = _compact_messages(
        messages, model_service,
        context_tokens=model_service.count_tokens(orchestrator_context or ""),
        max_context=max_ctx,
    )

    prompt = format_messages_to_prompt(
        messages,
        context=orchestrator_context,
        cwd=request.cwd,
        project_info=project_info,
        current_date=_request_current_date(request),
    )

    yield _status_event("preparing", "completed", "Preparing response...")

    # Store the user message in session history
    if request.session_id and last_user_msg:
        model_service.append_to_session(request.session_id, "user", last_user_msg)

    # `cancelled` is created earlier (before orchestration) and reused here so
    # the same signal covers both the planning phase and the response stream.

    def _is_degenerate(text: str) -> bool:
        """True when the model produced no meaningful content.

        A response of just ``\\`\\`\\``` (an unclosed code fence), whitespace,
        or fewer than 3 alphanumeric characters means the model tried to
        open a markdown block or stray token and hit EOS immediately.
        """
        stripped = text.strip().strip("`").strip("*").strip("_").strip()
        if len(stripped) < 3:
            return True
        alnum = sum(ch.isalnum() for ch in stripped)
        return alnum < 3

    def run_sync_stream():
        """Run the sync stream in a thread and put chunks in queue.

        When tool context is present we buffer only the OPENING of the
        stream so a degenerate output (small models often emit just ```)
        can be detected and replaced by a non-streaming retry.  Once we
        see enough real content we flush the buffer and stream normally
        so larger, capable models don't lose streaming UX.
        """
        collected: list[str] = []
        # Only buffer for tool-synthesis; regular chat streams immediately.
        buffered = orchestrator_context is not None
        opening_chars = 0
        _ALNUM_STREAM_THRESHOLD = 20

        try:
            for chunk in model_service.stream_completion(
                prompt, request.model, cancelled=cancelled,
                capability=response_capability,
            ):
                if cancelled.is_set():
                    break
                text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                if not text:
                    continue
                collected.append(text)

                if not buffered:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
                    continue

                # Still in the opening-buffer window.
                opening_chars += sum(ch.isalnum() for ch in text)
                if opening_chars >= _ALNUM_STREAM_THRESHOLD:
                    # Enough real content — flush buffered chunks and
                    # switch to normal streaming mode for the remainder.
                    for t in collected:
                        loop.call_soon_threadsafe(queue.put_nowait, t)
                    buffered = False
        except Exception as e:
            logger.error(f"Error in sync stream: {e}")
            loop.call_soon_threadsafe(queue.put_nowait, Exception(str(e)))
            loop.call_soon_threadsafe(queue.put_nowait, None)
            return

        if cancelled.is_set():
            loop.call_soon_threadsafe(queue.put_nowait, None)
            return

        response = "".join(collected)

        # If we already flushed the buffer we've been streaming — done.
        if not buffered:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            return

        if orchestrator_context and _is_degenerate(response):
            logger.warning(
                "[stream] Degenerate response (%r) — retrying with simplified prompt",
                response[:40],
            )
            # Strip the TOOL RESULTS wrapper and its rules, ask the model
            # to answer with a minimal prompt structure.  Same-prompt
            # retries produce the same output at low temperature, so we
            # only bother with an input that's actually different.
            raw_content = _extract_tool_content(orchestrator_context)
            retry = ""
            if raw_content:
                simple_prompt = (
                    f"<system>You are a helpful assistant.</system>\n"
                    f"<user>{_escape_role_tags(last_user_msg or 'Summarize the content below.')}\n\n"
                    f"Content:\n{_escape_role_tags(raw_content)}</user>"
                )
                try:
                    retry = model_service.generate_completion(
                        simple_prompt, request.model,
                        capability=response_capability,
                    )
                except Exception as e:
                    logger.error(f"[stream] Simplified retry failed: {e}")
                    retry = ""

            if retry and not _is_degenerate(retry):
                response = retry
            else:
                logger.warning(
                    "[stream] Simplified retry also degenerate (%r) — returning guidance message",
                    (retry or "")[:40],
                )
                response = (
                    "**No response generated.**\n\n"
                    "The current model wasn't able to synthesize the "
                    "fetched content. This usually means the loaded "
                    "model doesn't have the capabilities needed for "
                    "document reading or skill-based synthesis.\n\n"
                    "To fix this, open **Settings → Models** and enable "
                    "a model tagged with **DP** (document processing) or "
                    "**skills** capability. Larger knowledge-capable "
                    "models handle URL fetches, document Q&A, and "
                    "long-context summaries much more reliably than "
                    "the small default router model."
                )

        if orchestrator_context and response:
            # Emit the (possibly retried) full response as one chunk.
            loop.call_soon_threadsafe(queue.put_nowait, response)

        loop.call_soon_threadsafe(queue.put_nowait, None)

    # Start the sync stream in thread pool
    _executor.submit(run_sync_stream)

    # Signal that streaming has started
    yield _status_event("streaming", "processing", "Generating response...")

    # Initial chunk with role (OpenAI-compatible raw data format)
    message_id = request.message_id or f"msg-{uuid.uuid4().hex[:8]}"
    initial_chunk = ChatCompletionChunk(
        model=request.model,
        choices=[ChunkChoice(
            index=0,
            delta=DeltaContent(role="assistant"),
            finish_reason=None,
        )]
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    yield f"data: {json.dumps({'object': 'chat.message_id', 'message_id': message_id})}\n\n"

    # Stream content chunks and collect full response for session history
    full_response_parts = []
    try:
        while True:
            item = await queue.get()

            if item is None:
                # Stream complete
                break

            if isinstance(item, Exception):
                error_msg = str(item)
                logger.error(f"Stream error: {error_msg}")
                yield _status_event("error", "error", error_msg)
                # Surface the error as a visible content chunk so the user sees it
                error_chunk = ChatCompletionChunk(
                    model=request.model,
                    choices=[ChunkChoice(
                        index=0,
                        delta=DeltaContent(content=f"\n\n**Error:** {error_msg}"),
                        finish_reason=None,
                    )]
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                break

            full_response_parts.append(item)
            chunk = ChatCompletionChunk(
                model=request.model,
                choices=[ChunkChoice(
                    index=0,
                    delta=DeltaContent(content=item),
                    finish_reason=None,
                )]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
    finally:
        # Signal cancellation so the background thread stops producing chunks
        cancelled.set()

        # Save partial (or full) response to session history
        full_response = "".join(full_response_parts)
        logger.info(
            "[stream] Streaming finished — %d chunk(s), %d char(s), preview=%r",
            len(full_response_parts), len(full_response), full_response[:200],
        )
        if request.session_id and full_response:
            model_service.append_to_session(request.session_id, "assistant", full_response, citations=citations, message_id=message_id)

        # Response is fully streamed — free the on-demand auxiliary models
        # (vision + large agent) so only the orchestrator stays resident
        # between chats. Runs on normal completion, error, and cancellation.
        model_service.unload_auxiliary_models()

    # Auto file-write: if user asked to create/generate/write, save LLM response to file
    if full_response_parts and last_user_msg and request.cwd:
        full_response = "".join(full_response_parts)
        written_path = model_service.maybe_write_output(last_user_msg, full_response, request.cwd, session_id=request.session_id)
        if written_path:
            yield _status_event("file-write", "completed", written_path)
            # Surface the file path as visible content so the user can click it
            # Use forward slashes in the URL so the Markdown parser doesn't treat
            # backslashes as escape characters (e.g. \.out → .out on Windows).
            file_link = f"\n\n[{Path(written_path).name}]({Path(written_path).as_posix()})"
            write_chunk = ChatCompletionChunk(
                model=request.model,
                choices=[ChunkChoice(
                    index=0,
                    delta=DeltaContent(content=file_link),
                    finish_reason=None,
                )]
            )
            yield f"data: {write_chunk.model_dump_json()}\n\n"
            # Persist the file link in session history
            if request.session_id:
                model_service.append_to_session(request.session_id, "assistant", file_link)

    # Compute and emit usage info after streaming
    if full_response_parts:
        full_response = "".join(full_response_parts)
        prompt_tokens = model_service.count_tokens(prompt)
        completion_tokens = model_service.count_tokens(full_response)
        max_context = model_service.get_max_context_tokens()
        total_tokens = prompt_tokens + completion_tokens

        get_api_client().push_token_usage(
            project_id=project_id,
            model=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Emit usage info so the TUI can display context utilisation
        usage_payload = json.dumps({
            "object": "chat.usage",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "max_context_tokens": max_context,
        })
        yield f"data: {usage_payload}\n\n"

    if citations:
        # Deduplicate citations by document_id.  All cited documents
        # passed semantic relevance filtering in the discovery step,
        # so they are legitimate sources.  Previous revalidation
        # checked whether the filename appeared in the LLM response,
        # but that silently dropped documents whose filenames differ
        # from their content (e.g. "sp46.pdf" for a Vanguard fund).
        seen_ids: set = set()
        unique_citations = []
        for c in citations:
            if c["document_id"] in seen_ids:
                continue
            seen_ids.add(c["document_id"])
            unique_citations.append(c)
        if unique_citations:
            citations_payload = json.dumps({
                "object": "chat.citations",
                "citations": unique_citations,
            })
            yield f"data: {citations_payload}\n\n"

    yield _status_event("streaming", "completed", "Response complete")

    # Final chunk with finish_reason (OpenAI-compatible raw data format)
    final_chunk = ChatCompletionChunk(
        model=request.model,
        choices=[ChunkChoice(
            index=0,
            delta=DeltaContent(),
            finish_reason="stop",
        )]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """
    Create a chat completion (OpenAI-compatible).

    Supports both streaming and non-streaming modes.
    """
    try:
        if not model_service.is_loaded():
            logger.info("Model not loaded, attempting automatic reload...")
            try:
                model_service.load_model()
            except Exception as e:
                logger.error(f"Failed to auto-reload model: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="No model loaded and auto-reload failed. Call /v1/models/load first.",
                )

        if request.stream:
            return StreamingResponse(
                stream_chunks(request, model_service),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming response
        # Switch project if specified
        if request.project_id:
            try:
                model_service.set_current_project(request.project_id)
            except ValueError:
                logger.warning(f"Project not found: {request.project_id}, using current project")

        if not model_service.is_loaded():
            logger.info("Model not loaded (non-streaming), attempting automatic reload...")
            try:
                model_service.load_model()
            except Exception as e:
                logger.error(f"Failed to auto-reload model: {e}")
                raise RuntimeError(f"No model loaded and auto-reload failed: {e}")

        # Load the session's recent history ONCE and share it between the
        # prompt (prepended prior turns) and the orchestrator.
        _req_last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), None
        )
        conversation_history = model_service.get_relevant_history(
            request.session_id, _req_last_user or "",
        ) if request.session_id and _req_last_user else []

        messages = _build_messages_with_history(request, conversation_history)
        messages = _resolve_message_attachments(messages, model_service)

        last_user_msg = next(
            (m.content for m in reversed(messages) if m.role == "user"), None
        )

        project_skills = [s for s in model_service.get_project_skills() if not s.system]
        loop = asyncio.get_event_loop()

        _cur_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
        image_attachments = (
            model_service.get_asset_images(_cur_user.attachments)
            if _cur_user and _cur_user.attachments else []
        )

        # Orchestrator handles vector search, MCP tools, and explore in one step
        orchestrator_context = None
        tool_error = None
        citations = None
        is_mutation = False
        if last_user_msg:
            orchestrator_context, _, tool_error, citations, _, is_mutation = await loop.run_in_executor(
                _executor,
                lambda: model_service.orchestrate(
                    last_user_msg, skills=project_skills, cwd=request.cwd,
                    conversation_history=conversation_history,
                    images=image_attachments,
                    current_date=_request_current_date(request),
                ),
            )

        if tool_error:
            return ChatCompletionResponse(
                model=request.model,
                choices=[Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=tool_error),
                    finish_reason="stop",
                )],
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
        # For mutation operations return the success message directly.
        if is_mutation and orchestrator_context:
            return ChatCompletionResponse(
                model=request.model,
                choices=[Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=orchestrator_context),
                    finish_reason="stop",
                )],
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

        project_id = model_service.get_project_id()
        project_info = model_service.get_project_info()

        messages = _compact_messages(messages, model_service,
                                     context_tokens=model_service.count_tokens(orchestrator_context or ""))

        prompt = format_messages_to_prompt(
            messages,
            context=orchestrator_context,
            cwd=request.cwd,
            project_info=project_info,
            current_date=_request_current_date(request),
        )

        # Store user message in session history
        if request.session_id and last_user_msg:
            model_service.append_to_session(request.session_id, "user", last_user_msg)

        # Run sync completion in thread pool
        response_capability = ModelCapability.GENERAL if orchestrator_context else None
        response_text = await loop.run_in_executor(
            _executor,
            lambda: model_service.generate_completion(
                prompt, request.model, capability=response_capability,
            ),
        )

        # Auto file-write: if user asked to create/generate/write, save LLM response to file
        if last_user_msg and request.cwd and response_text:
            written_path = model_service.maybe_write_output(last_user_msg, response_text, request.cwd, session_id=request.session_id)
            if written_path:
                # Use forward slashes in the URL so the Markdown parser doesn't treat
                # backslashes as escape characters (e.g. \.out → .out on Windows).
                response_text = f"{response_text}\n\n[{Path(written_path).name}]({Path(written_path).as_posix()})"

        # Store assistant response in session history (after auto-write so the link is included)
        message_id = request.message_id or f"msg-{uuid.uuid4().hex[:8]}"
        if request.session_id and response_text:
            model_service.append_to_session(request.session_id, "assistant", response_text, citations=citations, message_id=message_id)

        prompt_tokens = model_service.count_tokens(prompt)
        completion_tokens = model_service.count_tokens(response_text)

        get_api_client().push_token_usage(
            project_id=project_id,
            model=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        citation_models = None
        if citations:
            seen_ids: set = set()
            unique_citations = []
            for c in citations:
                if c["document_id"] in seen_ids:
                    continue
                seen_ids.add(c["document_id"])
                unique_citations.append(c)
            if unique_citations:
                citation_models = [
                    Citation(document_id=c["document_id"], document_name=c.get("document_name"))
                    for c in unique_citations
                ]

        return ChatCompletionResponse(
            model=request.model,
            message_id=message_id,
            choices=[Choice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            citations=citation_models,
        )

    except Exception as e:
        logger.exception("Error in chat completion")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Free the on-demand auxiliary models (vision + large agent) after each
        # response so only the orchestrator stays resident. The streaming path
        # defers this to stream_chunks' finally (which fires when the stream
        # drains, not when the StreamingResponse object is returned here).
        if not request.stream:
            model_service.unload_auxiliary_models()
