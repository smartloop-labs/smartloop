"""
Planner — Classify → Plan pipeline for the MCP orchestration system.

Phase 1 (classify): Determines what kind of work a user message requires
    — tool call, built-in action, explore, multi-step, or conversation.
Phase 2 (create_plan): Builds a concrete step list from the classification.

The orchestrator handles Phase 3 (execute) and Phase 4 (return results).
"""

import json
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from smartloop.llms import BaseLLM

# Tool name constants — duplicated here to avoid circular imports with
# smartloop.agents.mcp.server_selector.
BASH_TOOL = "bash"
GREP_TOOL = "grep"
FILE_READ_TOOL = "file-read"
FILE_WRITE_TOOL = "file-write"
EXPLORE_TOOL = "explore"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# Models                                                                   #
# ---------------------------------------------------------------------- #

SYSTEM_TOOLS_SERVER_ID = "__system_tools__"
SYSTEM_TOOLS_SERVER_NAME = "smartloop"
FETCH_SERVER_ID = "__fetch__"


class ModelTier(str, Enum):
    """Which model tier a step should run on."""
    SMALL = "small"    # Routing, classification, arg extraction (1B)
    LARGE = "large"    # Skills, synthesis, complex reasoning (4B+)

class StepType(str, Enum):
    """Semantic action categories available in an execution plan."""
    ANALYZE        = "analyze"        # LLM reasons about user intent and context before planning
    TOOL           = "tool"           # Call an external MCP tool
    EXPLORE        = "explore"        # Walk the local filesystem tree + vector search
    ACTION         = "action"         # Built-in action: bash, write, read, grep, summarize
    CONVERSATIONAL = "conversational" # LLM generates content from knowledge


class IntentType(str, Enum):
    """High-level intent categories produced by classification."""
    TOOL_CALL    = "tool_call"       # Single MCP tool invocation
    ACTION       = "action"           # Single built-in action (bash, read, write, grep)
    EXPLORE      = "explore"          # Filesystem exploration + vector search
    MULTI_STEP   = "multi_step"       # 2+ steps needed
    CONVERSATION = "conversation"     # Pure LLM response, no tools


class Signal(str, Enum):
    """Deterministic signals detected in user input."""
    TOOLS = "tools"    # MCP tool keyword match
    PATHS = "paths"    # File or directory path(s)
    BASH  = "bash"     # Bash code block
    WRITE = "write"    # Write intent keywords
    GREP  = "grep"     # Grep/search keywords
    URL   = "url"      # HTTP(S) URL detected


class MatchedTool(BaseModel):
    """An MCP tool matched by keyword detection."""
    server_id: str
    server_name: str
    tool_name: str


class ClassificationResult(BaseModel):
    """Output of the classify phase — carries all detected signals."""
    intent: IntentType = IntentType.CONVERSATION
    detected_urls: List[str] = Field(default_factory=list)
    detected_paths: List[str] = Field(default_factory=list)
    matched_tools: List[MatchedTool] = Field(default_factory=list)
    has_write_intent: bool = False
    has_grep_intent: bool = False
    bash_commands: List[str] = Field(default_factory=list)
    action_kind: Optional[str] = None
    has_document_context: bool = False
    reasoning: str = ""
    capability: Optional[str] = Field(
        default=None,
        description="Best-matching model capability for this request (e.g. coding, finance, vision)",
    )


class PlanStep(BaseModel):
    """A single step in the LLM-generated execution plan."""
    step: int = Field(description="Step number (1-based)")
    step_type: StepType = Field(default=StepType.TOOL)
    instruction: str = Field(
        description=(
            "Detailed, self-contained instruction for this step — "
            "specifies what to look for, which file/URL to use, and what to do "
            "with the result."
        )
    )
    # For tool steps
    tool_name: Optional[str] = Field(default=None, description="Exact name of the tool to call")
    server_id: Optional[str] = Field(default=None, description="Server ID")
    server_name: Optional[str] = Field(default=None, description="Human-readable server name")
    # For action steps
    action_kind: Optional[str] = Field(
        default=None,
        description="Kind of built-in action: bash, write, read, grep, summarize",
    )
    # Conditional execution & provenance
    condition: Optional[str] = Field(
        default=None,
        description="Condition for execution: 'result_contains:X' or 'llm:question'",
    )
    skill_name: Optional[str] = Field(
        default=None,
        description="Name of the skill this step originated from",
    )
    # Fixed argument values specified in the skill body (e.g. team=DEV)
    default_args: Optional[Dict[str, str]] = Field(
        default=None,
        description="Default argument values from the skill definition",
    )
    # Bare argument names the LLM should extract from user input
    inferred_args: Optional[List[str]] = Field(
        default=None,
        description="Argument names to extract via LLM (from mcp block syntax)",
    )
    # Iterative tool calling — call the tool once per item from upstream results
    for_each: Optional[bool] = Field(
        default=None,
        description="When True, call the tool once per item extracted from upstream results",
    )
    model_tier: ModelTier = Field(
        default=ModelTier.SMALL,
        description="Which model tier this step should run on",
    )
    capability: Optional[str] = Field(
        default=None,
        description="Requested model capability for this step (e.g. coding, finance, vision)",
    )


class ExecutionPlan(BaseModel):
    """An LLM-generated ordered plan of tool calls to fulfil a user request."""
    steps: List[PlanStep] = Field(default_factory=list, description="Ordered steps to execute")
    reasoning: str = Field(default="", description="Brief explanation of why these steps were chosen")


class MCPToolContext(BaseModel):
    """Describes one MCP server and its tools for the planning prompt."""
    server_id: str
    server_name: str
    tools: List[Dict[str, Any]] = Field(default_factory=list)


class PlanContext(BaseModel):
    """Structured context assembled for the planner."""
    user_message: str
    skills: Optional[List[str]] = None
    mcp_tools: List[MCPToolContext] = Field(default_factory=list)
    document_context: Optional[str] = None
    cwd: Optional[str] = None
    detected_paths: List[str] = Field(default_factory=list)
    detected_urls: List[str] = Field(default_factory=list)
    skill_maps: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------- #
# Planner                                                                  #
# ---------------------------------------------------------------------- #

# Regex to find file/directory-like paths in user messages.
_PATH_RE = re.compile(
    r'(?:^|\s)([~/.][\w./\-]+|/[\w./\-]+)',
    re.MULTILINE,
)

# Regex to detect a ```bash ... ``` fenced code block in user text.
_BASH_BLOCK_RE = re.compile(r'```bash\s*\n(.+?)```', re.DOTALL)

# Multi-word phrases that signal file-write intent.
# Single words like "create" or "make" are too ambiguous (e.g. "create a recipe")
# and cause false positives — they're better handled by the LLM classifier.
_WRITE_KEYWORDS = {
    "write to", "write a file", "write file", "write it to",
    "save to", "save as", "save file", "save it to",
    "create file", "create a file",
    "rewrite",
}

# Keywords that signal a search/grep intent.
_GREP_KEYWORDS = {
    "search for", "grep", "find in files", "look for", "search across",
}

# URL extraction pattern (simple but effective).
_URL_RE = re.compile(r'https?://[^\s<>\'")\]]+')


class Planner:
    """Classifies user intent and creates execution plans."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    # ------------------------------------------------------------------ #
    # Phase 1: Classification                                              #
    # ------------------------------------------------------------------ #

    def classify(
        self,
        user_message: str,
        mcp_tools: List[MCPToolContext],
        skills: Optional[List[str]] = None,
        document_context: Optional[str] = None,
        has_attachments: bool = False,
        history: Optional[str] = None,
        cwd: Optional[str] = None,
        available_capabilities: Optional[List[str]] = None,
    ) -> ClassificationResult:
        """Classify user intent into one of the ``IntentType`` categories.

        Tier 1 (deterministic): Pattern-match URLs, paths, bash blocks,
        MCP tool keywords, write/grep intent.  No LLM call.

        Tier 2 (LLM fallback): Compact prompt when Tier 1 finds no signals.
        """
        result = self._detect_signals(user_message, mcp_tools, cwd, document_context, has_attachments)

        # Resolve intent from detected signals
        intent = self._resolve_intent(result)
        if intent is not None:
            result.intent = intent
            logger.info(
                f"[classify] deterministic → {intent.value} "
                f"(tools={[t.tool_name for t in result.matched_tools]}, "
                f"urls={len(result.detected_urls)}, paths={len(result.detected_paths)}, "
                f"write={result.has_write_intent}, grep={result.has_grep_intent}, "
                f"bash={len(result.bash_commands)})"
            )
        else:
            # Tier 2: LLM fallback
            result = self._llm_classify(user_message, mcp_tools, skills, history, result)
            logger.info(f"[classify] LLM → {result.intent.value}: {result.reasoning}")

        if available_capabilities:
            result.capability = self._match_capability(
                available_capabilities,
                has_attachments=has_attachments,
                has_matched_skills=bool(skills),
                has_document_context=result.has_document_context,
            )
            if result.capability:
                logger.info(f"[classify] capability → {result.capability}")

        return result

    @staticmethod
    def _match_capability(
        capabilities: List[str],
        *,
        has_attachments: bool = False,
        has_matched_skills: bool = False,
        has_document_context: bool = False,
    ) -> Optional[str]:
        """Pick a capability from planning signals in priority order.

        Priority:
          1. ``vision``    — attachments imply image/document OCR work.
          2. ``skills``    — a matched skill defines the procedure to run.
          3. ``dp``        — indexed document context: data-processing model.
          4. ``knowledge`` — default for open-ended chat / general answers.

        Returns the first capability whose signal fires *and* is present
        in ``capabilities``, or ``None`` when nothing matches.
        """
        priority = (
            ("vision", has_attachments),
            ("skills", has_matched_skills),
            ("dp", has_document_context),
            ("knowledge", True),
        )
        for cap, signal in priority:
            if signal and cap in capabilities:
                return cap
        return None

    def _detect_signals(
        self,
        user_message: str,
        mcp_tools: List[MCPToolContext],
        cwd: Optional[str] = None,
        document_context: Optional[str] = None,
        has_attachments: bool = False,
    ) -> ClassificationResult:
        """Extract all deterministic signals from the user message."""
        result = ClassificationResult()
        msg_lower = user_message.lower()

        # MCP tool keyword match
        for srv in mcp_tools:
            for tool in srv.tools:
                t_name = tool.get("name", "")
                snake = re.split(r'[_\-]', t_name)
                camel_parts: List[str] = []
                for part in snake:
                    camel_parts.extend(re.sub(r'([A-Z])', r' \1', part).split())
                words = [w.lower() for w in camel_parts if len(w) > 2]
                if words and all(w in msg_lower for w in words):
                    result.matched_tools.append(MatchedTool(
                        server_id=srv.server_id,
                        server_name=srv.server_name,
                        tool_name=t_name,
                    ))
                    break  # one match per server

        # URLs
        result.detected_urls = _URL_RE.findall(user_message)

        # File / directory paths
        if cwd:
            result.detected_paths = _extract_paths(user_message, cwd)

        # Bash blocks
        for m in _BASH_BLOCK_RE.finditer(user_message):
            result.bash_commands.append(m.group(1).strip())

        # Write intent
        result.has_write_intent = any(kw in msg_lower for kw in _WRITE_KEYWORDS)

        # Grep intent
        result.has_grep_intent = any(kw in msg_lower for kw in _GREP_KEYWORDS)

        # Document context from vector search / uploads / attachments
        result.has_document_context = has_attachments or bool(document_context and document_context.strip())

        return result

    def _resolve_intent(self, result: ClassificationResult) -> Optional[IntentType]:
        """Derive intent from detected signals.  Returns None when ambiguous."""
        # Build set of active signals
        active: set[Signal] = set()
        if result.matched_tools:
            active.add(Signal.TOOLS)
        if result.detected_paths:
            active.add(Signal.PATHS)
        if result.bash_commands:
            active.add(Signal.BASH)
        if result.has_write_intent:
            active.add(Signal.WRITE)
        if result.has_grep_intent:
            active.add(Signal.GREP)
        if result.detected_urls:
            active.add(Signal.URL)

        if not active:
            # Document context present → treat as EXPLORE so the model
            # answers from the attached/uploaded content instead of
            # falling through to the LLM classifier (which often picks
            # CONVERSATION and ignores the documents).
            if result.has_document_context:
                result.reasoning = "document context available — exploring"
                return IntentType.EXPLORE
            return None

        # Action signals (exclude modifiers like write/grep)
        action_signals = active & {Signal.TOOLS, Signal.PATHS, Signal.BASH, Signal.URL}

        # Multiple action signal types → multi-step
        if len(action_signals) > 1:
            result.reasoning = "multiple action signals detected"
            return IntentType.MULTI_STEP

        # URL + write modifier → fetch then write
        if Signal.URL in active and Signal.WRITE in active:
            result.reasoning = "URL + write intent"
            return IntentType.MULTI_STEP

        # Single tool + write/grep modifier → still multi-step
        if Signal.TOOLS in active and (active & {Signal.WRITE, Signal.GREP}):
            # MCP tool match with write intent is still just the tool call
            # (the MCP tool IS the action, e.g. "create issue" matches write keyword)
            pass
        elif Signal.PATHS in active and Signal.WRITE in active:
            result.reasoning = "file paths + write intent"
            return IntentType.MULTI_STEP

        # Single signal type
        if Signal.URL in active:
            result.reasoning = f"URL detected: {result.detected_urls[0]}"
            return IntentType.TOOL_CALL

        if Signal.TOOLS in active:
            result.reasoning = f"MCP tool keyword match: {result.matched_tools[0].tool_name}"
            return IntentType.TOOL_CALL

        if Signal.BASH in active:
            result.action_kind = "bash"
            result.reasoning = "bash code block detected"
            return IntentType.ACTION

        if Signal.PATHS in active:
            # Check if any path is a directory → explore; otherwise read
            dirs = [p for p in result.detected_paths if os.path.isdir(p)]
            if dirs:
                result.reasoning = f"directory path detected: {dirs[0]}"
                return IntentType.EXPLORE
            result.action_kind = "read"
            result.reasoning = f"file path detected: {result.detected_paths[0]}"
            return IntentType.ACTION

        if Signal.GREP in active:
            result.action_kind = "grep"
            result.reasoning = "grep/search keywords detected"
            return IntentType.ACTION

        if Signal.WRITE in active:
            # Write intent alone (no paths, no URLs) — needs content generation + write
            result.reasoning = "write intent without target"
            return IntentType.MULTI_STEP

        return None  # shouldn't reach here, but safe fallback

    def _llm_classify(
        self,
        user_message: str,
        mcp_tools: List[MCPToolContext],
        skills: Optional[List[str]],
        history: Optional[str],
        result: ClassificationResult,
    ) -> ClassificationResult:
        """Lightweight LLM call to classify ambiguous messages."""
        # Build compact skill/server list
        skill_lines = ""
        if skills:
            skill_lines = "Skills:\n" + "\n".join(
                f"  - {s[:200]}" for s in skills[:5]
            ) + "\n"

        server_lines = ""
        if mcp_tools:
            lines = []
            for srv in mcp_tools:
                tool_names = [t.get("name", "") for t in srv.tools[:5]]
                lines.append(f"  {srv.server_name}: {', '.join(tool_names)}")
            server_lines = "MCP servers:\n" + "\n".join(lines) + "\n"

        history_line = ""
        if history:
            history_line = f"Recent context: {history[:600]}\n"

        doc_line = ""
        if result.has_document_context:
            doc_line = (
                "Document context is available from uploaded documents.\n"
                "Use EXPLORE only if the question is specifically about the user's documents.\n"
                "Use CONVERSATION if the question is general knowledge (e.g. recommendations, facts, opinions).\n"
            )

        prompt = (
            "<system>Classify the user's intent. Reply with exactly one word:\n"
            "TOOL_CALL — needs an external MCP tool (API call, fetch, etc.)\n"
            "ACTION — needs a local action (read file, run bash, grep, write file)\n"
            "EXPLORE — needs to browse/explore local files or search uploaded documents\n"
            "MULTI_STEP — needs multiple steps\n"
            "CONVERSATION — can be answered from general knowledge or general conversation "
            "(e.g. recommendations, trivia, opinions, greetings)\n\n"
            f"{doc_line}"
            f"{skill_lines}"
            f"{server_lines}"
            f"{history_line}"
            "</system>"
            f"<user>{user_message}</user>"
        )

        try:
            # Ensure the prompt fits
            max_tokens = self.llm.get_max_tokens()
            token_budget = int(max_tokens * 0.8)
            prompt_tokens = self.llm.get_num_tokens(prompt)
            if prompt_tokens > token_budget:
                # Drop skills and history to fit
                prompt = (
                    "<system>Classify the user's intent. Reply with exactly one word:\n"
                    "TOOL_CALL, ACTION, EXPLORE, MULTI_STEP, or CONVERSATION</system>"
                    f"<user>{user_message}</user>"
                )

            response = str(self.llm.invoke(prompt)).strip().upper()
            # Extract the intent word from the response
            intent_map = {
                "TOOL_CALL": IntentType.TOOL_CALL,
                "ACTION": IntentType.ACTION,
                "EXPLORE": IntentType.EXPLORE,
                "DOCUMENT": IntentType.EXPLORE,
                "MULTI_STEP": IntentType.MULTI_STEP,
                "CONVERSATION": IntentType.CONVERSATION,
            }
            for key, intent in intent_map.items():
                if key in response:
                    result.intent = intent
                    result.reasoning = f"LLM classified as {key}"
                    return result

            # Could not parse → default to conversation
            result.intent = IntentType.CONVERSATION
            result.reasoning = f"LLM response unparseable ({response[:50]}), defaulting to conversation"
            return result

        except Exception as e:
            logger.warning(f"[classify] LLM fallback failed: {e}")
            result.intent = IntentType.CONVERSATION
            result.reasoning = f"LLM classification failed: {e}"
            return result

    # ------------------------------------------------------------------ #
    # Phase 2: Plan Creation                                               #
    # ------------------------------------------------------------------ #

    def build_context(
        self,
        user_message: str,
        available_servers: List[Dict[str, Any]],
        cwd: Optional[str] = None,
        skills: Optional[List[str]] = None,
        document_context: Optional[str] = None,
    ) -> PlanContext:
        """Assemble a ``PlanContext`` from raw inputs."""
        mcp_tools: List[MCPToolContext] = []
        for srv in available_servers:
            if srv.get("id") in (SYSTEM_TOOLS_SERVER_ID, FETCH_SERVER_ID):
                continue
            mcp_tools.append(MCPToolContext(
                server_id=srv.get("id", ""),
                server_name=srv.get("name", "Unknown"),
                tools=srv.get("tools", []),
            ))

        detected_paths = _extract_paths(user_message, cwd) if cwd else []
        detected_urls = _URL_RE.findall(user_message)

        return PlanContext(
            user_message=user_message,
            skills=skills,
            mcp_tools=mcp_tools,
            document_context=document_context,
            cwd=cwd,
            detected_paths=detected_paths,
            detected_urls=detected_urls,
        )

    def create_plan(
        self,
        classification: ClassificationResult,
        context: PlanContext,
    ) -> ExecutionPlan:
        """Build an execution plan from a classification result.

        For CONVERSATION with a matched capability (e.g. ``knowledge``) →
        one CONVERSATIONAL step routed to that capability's model.
        For CONVERSATION without a capability → empty plan (router streams
        via the small model directly).
        For TOOL_CALL, ACTION, EXPLORE → single-step plan (no LLM call).
        For MULTI_STEP → LLM-generated plan with fallback to signal-based steps.
        """
        intent = classification.intent

        if intent == IntentType.CONVERSATION:
            if classification.capability:
                return ExecutionPlan(
                    steps=[PlanStep(
                        step=1,
                        step_type=StepType.CONVERSATIONAL,
                        tool_name="conversation",
                        server_id=SYSTEM_TOOLS_SERVER_ID,
                        server_name=SYSTEM_TOOLS_SERVER_NAME,
                        instruction=context.user_message,
                        model_tier=ModelTier.LARGE,
                        capability=classification.capability,
                    )],
                    reasoning=f"conversation → capability '{classification.capability}'",
                )
            return ExecutionPlan(steps=[], reasoning="conversation (no capability)")

        if intent == IntentType.TOOL_CALL:
            return self._plan_tool_call(classification, context)

        if intent == IntentType.ACTION:
            return self._plan_action(classification, context)

        if intent == IntentType.EXPLORE:
            return self._plan_explore(classification, context)

        if intent == IntentType.MULTI_STEP:
            return self._plan_multi_step(classification, context)

        return ExecutionPlan(steps=[], reasoning="unknown intent")

    def _plan_tool_call(
        self, classification: ClassificationResult, context: PlanContext,
    ) -> ExecutionPlan:
        """Single MCP tool step."""
        if classification.matched_tools:
            tool = classification.matched_tools[0]
            return ExecutionPlan(
                steps=[PlanStep(
                    step=1, step_type=StepType.TOOL,
                    tool_name=tool.tool_name,
                    server_id=tool.server_id,
                    server_name=tool.server_name,
                    instruction=context.user_message,
                )],
                reasoning=f"MCP tool: {tool.tool_name}",
            )

        # URL fetch — find a fetch-capable tool
        if classification.detected_urls:
            url = classification.detected_urls[0]
            fetch_step = self._make_fetch_step(url, context)
            if fetch_step:
                fetch_step.step = 1
                return ExecutionPlan(
                    steps=[fetch_step],
                    reasoning=f"fetch URL: {url}",
                )
            # No fetch tool found — placeholder for orchestrator to resolve
            return ExecutionPlan(
                steps=[PlanStep(
                    step=1, step_type=StepType.TOOL,
                    tool_name="fetch",
                    server_id="", server_name="",
                    instruction=f"Fetch {url} and extract its main content.",
                )],
                reasoning=f"fetch URL (unresolved): {url}",
            )

        return ExecutionPlan(steps=[], reasoning="tool_call but no tool matched")

    def _plan_action(
        self, classification: ClassificationResult, context: PlanContext,
    ) -> ExecutionPlan:
        """Single built-in system tool step.

        System tools run as ordinary TOOL steps against the local
        ``smartloop`` MCP server (like fetch).  Where the classifier already
        extracted a concrete value (bash command, file path) it is carried as
        ``default_args`` so resolution stays deterministic; otherwise the
        argument extractor fills the args from the user message.
        """
        action_kind = classification.action_kind or "read"

        if action_kind == "bash" and classification.bash_commands:
            cmd = classification.bash_commands[0]
            return ExecutionPlan(
                steps=[PlanStep(
                    step=1, step_type=StepType.TOOL,
                    tool_name=BASH_TOOL,
                    server_id=SYSTEM_TOOLS_SERVER_ID,
                    server_name=SYSTEM_TOOLS_SERVER_NAME,
                    instruction=f"Run: {cmd}",
                    default_args={"command": cmd},
                )],
                reasoning="bash command",
            )

        if action_kind == "read" and classification.detected_paths:
            path = classification.detected_paths[0]
            return ExecutionPlan(
                steps=[PlanStep(
                    step=1, step_type=StepType.TOOL,
                    tool_name=FILE_READ_TOOL,
                    server_id=SYSTEM_TOOLS_SERVER_ID,
                    server_name=SYSTEM_TOOLS_SERVER_NAME,
                    instruction=f"Read the file {path}",
                    default_args={"path": path},
                )],
                reasoning=f"read file: {path}",
            )

        if action_kind == "grep":
            return ExecutionPlan(
                steps=[PlanStep(
                    step=1, step_type=StepType.TOOL,
                    tool_name=GREP_TOOL,
                    server_id=SYSTEM_TOOLS_SERVER_ID,
                    server_name=SYSTEM_TOOLS_SERVER_NAME,
                    instruction="Search files for the requested pattern.",
                )],
                reasoning="grep search",
            )

        # Generic system-tool fallback
        return ExecutionPlan(
            steps=[PlanStep(
                step=1, step_type=StepType.TOOL,
                tool_name={
                    "read": FILE_READ_TOOL,
                    "write": FILE_WRITE_TOOL,
                    "grep": GREP_TOOL,
                    "bash": BASH_TOOL,
                }.get(action_kind, FILE_READ_TOOL),
                server_id=SYSTEM_TOOLS_SERVER_ID,
                server_name=SYSTEM_TOOLS_SERVER_NAME,
                instruction=context.user_message,
            )],
            reasoning=f"action: {action_kind}",
        )

    def _plan_explore(
        self, classification: ClassificationResult, context: PlanContext,
    ) -> ExecutionPlan:
        """Single explore step — execution handles fetch → vector → files internally."""
        target = classification.detected_paths[0] if classification.detected_paths else (context.cwd or ".")
        return ExecutionPlan(
            steps=[PlanStep(
                step=1, step_type=StepType.EXPLORE,
                tool_name=EXPLORE_TOOL,
                server_id=SYSTEM_TOOLS_SERVER_ID,
                server_name=SYSTEM_TOOLS_SERVER_NAME,
                instruction=f"Explore directory {target}",
            )],
            reasoning=f"explore: {target}",
        )

    def _plan_multi_step(
        self, classification: ClassificationResult, context: PlanContext,
    ) -> ExecutionPlan:
        """Build a multi-step plan.

        Tries the LLM for step ordering; falls back to building steps
        directly from the classification's detected signals.
        """
        # Try LLM-based planning
        plan = self._llm_plan(classification, context)
        if plan.steps:
            return plan

        # Fallback: build steps from signals
        return self._build_steps_from_signals(classification, context)

    def _llm_plan(
        self, classification: ClassificationResult, context: PlanContext,
    ) -> ExecutionPlan:
        """Call LLM to produce a step ordering for multi-step requests."""
        # Build a focused prompt — classification already tells us what signals exist
        signals_hint = []
        if classification.matched_tools:
            for t in classification.matched_tools:
                signals_hint.append(f"MCP tool available: {t.tool_name} (server: {t.server_name})")
        if classification.detected_urls:
            signals_hint.append(f"URLs: {', '.join(classification.detected_urls[:3])}")
        if classification.detected_paths:
            signals_hint.append(f"Paths: {', '.join(classification.detected_paths[:3])}")
        if classification.has_write_intent:
            signals_hint.append("Write intent detected")
        if classification.has_grep_intent:
            signals_hint.append("Grep/search intent detected")
        if classification.bash_commands:
            signals_hint.append(f"Bash: {classification.bash_commands[0][:100]}")

        # Skills section
        skills_section = ""
        if context.skills:
            skills_section = (
                "SKILLS (use these to decide which server to call):\n"
                + "\n".join(f"  - {s}" for s in context.skills)
                + "\n\n"
            )

        # Skill maps section (explicit step skeletons from skill_map)
        skill_maps_section = ""
        if context.skill_maps:
            sm_lines: List[str] = []
            for sm in context.skill_maps:
                sm_lines.append(f"  Skill: {sm['skill_name']}")
                for step_entry in sm.get("steps", []):
                    parts = [f"    step {step_entry.get('step', '?')}"]
                    if step_entry.get("tool"):
                        parts.append(f"tool={step_entry['tool']}")
                    if step_entry.get("server"):
                        parts.append(f"server={step_entry['server']}")
                    if step_entry.get("condition"):
                        parts.append(f"condition=\"{step_entry['condition']}\"")
                    if step_entry.get("instruction"):
                        parts.append(f"— {step_entry['instruction']}")
                    sm_lines.append(" ".join(parts))
            skill_maps_section = (
                "SKILL STEPS (use these as step skeleton — preserve tool, server, and condition):\n"
                + "\n".join(sm_lines) + "\n\n"
            )

        # Servers section — include actual tool names so the LLM uses exact
        # names instead of guessing / inventing non-existent ones.
        server_lines: List[str] = []
        for srv in context.mcp_tools:
            tool_names = [t.get("name", "") for t in srv.tools[:10] if t.get("name")]
            tools_list = ", ".join(tool_names) if tool_names else "(no tools)"
            server_lines.append(
                f'  "{srv.server_name}" (server_id="{srv.server_id}"): '
                f'tools=[{tools_list}]'
            )
        servers_section = (
            "MCP SERVERS (use ONLY these exact tool names):\n"
            + "\n".join(server_lines) + "\n\n"
        ) if server_lines else ""

        prompt = (
            "<system>"
            "You are a task planner. Return a JSON array of steps.\n\n"
            f"{skills_section}"
            f"{skill_maps_section}"
            f"{servers_section}"
            "SYSTEM TOOLS (server_id='__system_tools__'):\n"
            "  explore — walk the local file tree\n"
            "  file-read — read a local file\n"
            "  file-write — write a local file\n"
            "  grep — search text in local files\n"
            "  bash — run a shell command\n\n"
            "DETECTED SIGNALS:\n"
            + "\n".join(f"  - {s}" for s in signals_hint)
            + "\n\n"
            "STEP FORMAT:\n"
            '{"step":<n>,"step_type":"tool|action|explore","tool_name":"<name>",'
            '"server_id":"<id>","server_name":"<name>","instruction":"<what to do>",'
            '"action_kind":"read|write|grep|bash (action steps only)",'
            '"condition":"result_contains:X or llm:question (optional)",'
            '"skill_name":"originating skill name (optional)"}\n\n'
            "RULES:\n"
            "  1. tool_name MUST be an exact name from the MCP SERVERS or SYSTEM TOOLS lists above. "
            "Never invent or guess tool names.\n"
            "  2. If a skill matches the request → step_type='tool', use the server from skills.\n"
            "  3. For local file operations → step_type='action' with system tools.\n"
            "  4. Max 5 steps.\n"
            "  5. If SKILL STEPS are provided, preserve their tool, server, and condition values.\n"
            "  6. If skill instructions contain conditional language (e.g. 'if there are differences', "
            "'only when', 'in case of'), emit a 'condition' field on the conditional step:\n"
            "     - Use 'result_contains:<keyword>' for simple text checks on previous results.\n"
            "     - Use 'llm:<yes/no question>' for complex conditions requiring reasoning.\n"
            "  7. Each step can use results from previous steps. Write instructions that reference "
            "prior output (e.g. 'Using the invoice summary from step 1, ...').\n"
            "  8. NEVER use file-read to read uploaded documents (PDFs, DOCX, etc.). "
            "They are already vectorized and available via the document context. "
            "Use step_type='explore' if you need to search/analyze documents.\n"
            "</system>"
            "<user>"
            f"{context.user_message}\n\n"
            "Reply with ONLY the JSON array."
            "</user>"
        )

        # Budget check
        max_tokens = self.llm.get_max_tokens()
        token_budget = int(max_tokens * 0.8)
        prompt_tokens = self.llm.get_num_tokens(prompt)
        if prompt_tokens > token_budget:
            # Trim skills
            if context.skills:
                context = context.model_copy(update={"skills": None})
                # Rebuild prompt without skills
                prompt = (
                    "<system>You are a task planner. Return a JSON array of steps.\n\n"
                    f"{servers_section}"
                    "DETECTED SIGNALS:\n"
                    + "\n".join(f"  - {s}" for s in signals_hint) + "\n\n"
                    "STEP FORMAT:\n"
                    '{"step":<n>,"step_type":"tool|action|explore","tool_name":"<name>",'
                    '"server_id":"<id>","server_name":"<name>","instruction":"<what to do>",'
                    '"action_kind":"read|write|grep|bash (action steps only)"}\n\n'
                    "</system>"
                    f"<user>{context.user_message}\n\nReply with ONLY the JSON array.</user>"
                )

        try:
            response = str(self.llm.invoke(prompt)).strip()
            logger.info(f"[plan] Raw LLM plan ({len(response)} chars): {response[:500]}")
            return self._parse_plan_response(response, context.user_message, classification)
        except Exception as e:
            logger.warning(f"[plan] LLM plan failed: {e}")
            return ExecutionPlan(steps=[], reasoning=str(e))

    def _build_steps_from_signals(
        self, classification: ClassificationResult, context: PlanContext,
    ) -> ExecutionPlan:
        """Build plan steps directly from classification signals (no LLM)."""
        steps: List[PlanStep] = []
        step_num = 1

        # Fetch URLs first (context gathering)
        for url in classification.detected_urls:
            fetch_step = self._make_fetch_step(url, context)
            if fetch_step:
                fetch_step.step = step_num
                steps.append(fetch_step)
            else:
                steps.append(PlanStep(
                    step=step_num, step_type=StepType.TOOL,
                    tool_name="fetch", server_id="", server_name="",
                    instruction=f"Fetch {url} and extract its main content.",
                ))
            step_num += 1

        # MCP tool calls
        for tool in classification.matched_tools:
            steps.append(PlanStep(
                step=step_num, step_type=StepType.TOOL,
                tool_name=tool.tool_name,
                server_id=tool.server_id,
                server_name=tool.server_name,
                instruction=context.user_message,
            ))
            step_num += 1

        # File paths → read or explore
        for path in classification.detected_paths:
            if os.path.isdir(path):
                steps.append(PlanStep(
                    step=step_num, step_type=StepType.EXPLORE,
                    tool_name=EXPLORE_TOOL,
                    server_id=SYSTEM_TOOLS_SERVER_ID,
                    server_name=SYSTEM_TOOLS_SERVER_NAME,
                    instruction=f"Explore directory {path}",
                ))
            else:
                steps.append(PlanStep(
                    step=step_num, step_type=StepType.TOOL,
                    tool_name=FILE_READ_TOOL,
                    server_id=SYSTEM_TOOLS_SERVER_ID,
                    server_name=SYSTEM_TOOLS_SERVER_NAME,
                    instruction=f"Read the file {path}",
                    default_args={"path": path},
                ))
            step_num += 1

        # Bash blocks
        for cmd in classification.bash_commands:
            steps.append(PlanStep(
                step=step_num, step_type=StepType.TOOL,
                tool_name=BASH_TOOL,
                server_id=SYSTEM_TOOLS_SERVER_ID,
                server_name=SYSTEM_TOOLS_SERVER_NAME,
                instruction=f"Run: {cmd}",
                default_args={"command": cmd},
            ))
            step_num += 1

        # Grep
        if classification.has_grep_intent:
            steps.append(PlanStep(
                step=step_num, step_type=StepType.TOOL,
                tool_name=GREP_TOOL,
                server_id=SYSTEM_TOOLS_SERVER_ID,
                server_name=SYSTEM_TOOLS_SERVER_NAME,
                instruction="Search files for the requested pattern.",
            ))
            step_num += 1

        # Write intent → conversational generation + file write
        if classification.has_write_intent:
            steps.append(PlanStep(
                step=step_num, step_type=StepType.CONVERSATIONAL,
                tool_name="conversation",
                server_id=SYSTEM_TOOLS_SERVER_ID,
                server_name=SYSTEM_TOOLS_SERVER_NAME,
                instruction="Generate the requested content based on the user request and any fetched context.",
            ))
            step_num += 1
            steps.append(PlanStep(
                step=step_num, step_type=StepType.TOOL,
                tool_name=FILE_WRITE_TOOL,
                server_id=SYSTEM_TOOLS_SERVER_ID,
                server_name=SYSTEM_TOOLS_SERVER_NAME,
                instruction="Write the generated content to a file.",
            ))
            step_num += 1

        reasoning = f"signal-based: {classification.reasoning}"
        return ExecutionPlan(steps=steps[:5], reasoning=reasoning)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    # Keywords that identify a URL-fetching tool by name or description.
    _FETCH_TOOL_KEYWORDS = {"fetch", "browse", "scrape", "web", "http", "navigate", "get_url"}

    def _make_fetch_step(self, url: str, context: PlanContext) -> Optional[PlanStep]:
        """Return None so the orchestrator resolves fetch to the right server.

        The local fetch server (``__fetch__``) is excluded from
        ``context.mcp_tools``, so matching here would only find remote
        servers.  Returning ``None`` lets the orchestrator's
        ``_resolve_fetch_steps`` wire the step to the correct
        (local-first) fetch tool at execution time.
        """
        return None

    # ------------------------------------------------------------------ #
    # Response parsing                                                     #
    # ------------------------------------------------------------------ #

    def _parse_plan_response(
        self,
        response: str,
        user_message: str,
        classification: Optional[ClassificationResult] = None,
    ) -> ExecutionPlan:
        """Parse the raw LLM response into an ``ExecutionPlan``."""
        # Strip markdown fences
        if response.startswith("```"):
            lines = response.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            response = "\n".join(lines).strip()

        try:
            steps_data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"[plan] JSON parse error: {e}")
            return ExecutionPlan(steps=[], reasoning=f"Parse error: {e}")

        if not isinstance(steps_data, list):
            logger.warning("[plan] Plan response is not a JSON array")
            return ExecutionPlan(steps=[], reasoning="Parse error: not a list")

        if not steps_data:
            logger.info("[plan] Planner returned empty plan — treating as conversation")
            return ExecutionPlan(steps=[], reasoning="conversation")

        steps: List[PlanStep] = []
        for i, s in enumerate(steps_data[:5]):  # hard cap at 5 steps
            if not isinstance(s, dict):
                continue

            raw_step_type = s.get("step_type", "")
            # Backward compatibility: map old step_type values. System tools
            # (read/write/grep/bash) now run as ordinary TOOL steps against
            # the local smartloop MCP server, like fetch.
            step_type_map = {
                "tool_call": StepType.TOOL,
                "mcp": StepType.TOOL,
                "conversation": StepType.CONVERSATIONAL,
                "document": StepType.EXPLORE,
                "read": StepType.TOOL,
                "write": StepType.TOOL,
                "grep": StepType.TOOL,
                "bash": StepType.TOOL,
            }

            if raw_step_type in step_type_map:
                step_type = step_type_map[raw_step_type]
            else:
                try:
                    step_type = StepType(raw_step_type)
                except ValueError:
                    step_type = self._infer_step_type(
                        s.get("tool_name", ""),
                        s.get("server_id", SYSTEM_TOOLS_SERVER_ID),
                    )

            # Correct misclassified steps: if the LLM said "action" or
            # "conversational" but the server_id is NOT __system_tools__,
            # it's an external MCP tool call — override to TOOL.
            t_server_id = s.get("server_id", SYSTEM_TOOLS_SERVER_ID)
            if (step_type in (StepType.ACTION, StepType.CONVERSATIONAL)
                    and t_server_id != SYSTEM_TOOLS_SERVER_ID
                    and s.get("tool_name")):
                step_type = StepType.TOOL

            # The built-in file/grep/bash tools resolve through the smartloop
            # MCP server, so they run as TOOL steps pinned to that server —
            # even if the model mislabelled the step as "action". Only
            # ``explore`` is an internal pseudo-step.
            t_name = s.get("tool_name", "")
            if t_name == EXPLORE_TOOL:
                step_type = StepType.EXPLORE
                t_server_id = SYSTEM_TOOLS_SERVER_ID
            elif t_name in (FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL, BASH_TOOL):
                step_type = StepType.TOOL
                t_server_id = SYSTEM_TOOLS_SERVER_ID

            # Infer action_kind for the remaining ACTION steps (summarize).
            action_kind = s.get("action_kind")

            steps.append(PlanStep(
                step=s.get("step", i + 1),
                step_type=step_type,
                tool_name=s.get("tool_name"),
                server_id=t_server_id if t_server_id else s.get("server_id", SYSTEM_TOOLS_SERVER_ID),
                server_name=s.get("server_name", SYSTEM_TOOLS_SERVER_NAME),
                instruction=s.get("instruction") or user_message,
                action_kind=action_kind,
                condition=s.get("condition") or None,
                skill_name=s.get("skill_name") or None,
            ))

        if classification:
            path_idx = 0
            cmd_idx = 0
            for step in steps:
                if step.default_args:
                    continue
                if step.tool_name == FILE_READ_TOOL and classification.detected_paths:
                    if path_idx < len(classification.detected_paths):
                        step.default_args = {"path": classification.detected_paths[path_idx]}
                        path_idx += 1
                elif step.tool_name == BASH_TOOL and classification.bash_commands:
                    if cmd_idx < len(classification.bash_commands):
                        step.default_args = {"command": classification.bash_commands[cmd_idx]}
                        cmd_idx += 1

        logger.info(
            f"[plan] Created {len(steps)}-step plan: "
            f"{[f'{s.step}:{s.step_type.value}:{s.tool_name}' for s in steps]}"
        )
        return ExecutionPlan(steps=steps)

    @staticmethod
    def _infer_step_type(tool_name: str, server_id: str) -> StepType:
        """Derive ``StepType`` from ``tool_name`` and ``server_id``.

        ``explore`` is the only internal pseudo-step; the other built-in
        tools (file/grep/bash) run as ordinary TOOL steps against the local
        smartloop MCP server.
        """
        if tool_name == EXPLORE_TOOL:
            return StepType.EXPLORE
        return StepType.TOOL


# ---------------------------------------------------------------------- #
# Module-level helpers                                                     #
# ---------------------------------------------------------------------- #

def _extract_paths(text: str, cwd: str) -> List[str]:
    """Return real filesystem paths found in *text*."""
    found: List[str] = []
    for m in _PATH_RE.finditer(text):
        raw = m.group(1).rstrip(".,;:!?")
        expanded = os.path.expanduser(raw)
        if not os.path.isabs(expanded):
            expanded = os.path.join(cwd, expanded)
        expanded = os.path.normpath(expanded)
        if os.path.exists(expanded) and expanded not in found:
            found.append(expanded)
    return found
