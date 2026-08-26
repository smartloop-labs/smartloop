"""Skills: matching them to a request, and shaping their output.

A skill is a markdown document the user wrote describing a procedure.  This
module covers both ends of using one: picking which skill (if any) a request
maps to, and turning the steps' raw output back into the answer the skill's
body describes.

Everything in between — planning and executing the steps — is the pipeline's
job.  A matched skill contributes guiding context to the planner; it does not
get its own control flow.
"""
import logging
import re
from typing import Dict, List, Optional

from smartloop.agents.nodes import NodeResult, SkillMatchNode
from smartloop.agents.mcp.explorer import Explorer
from smartloop.agents.mcp.server_selector import (
    BASH_TOOL, EXPLORE_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL,
)
from smartloop.agents.orchestration.context import _OrchestratorContext
from smartloop.agents.orchestration.model_router import (
    DECISION_MAX_TOKENS, SHORT_TEXT_MAX_TOKENS, ModelRouter,
)
from smartloop.agents.orchestration.response_builder import TOOL_CODE_FENCE_RE
from smartloop.agents.planner import ModelTier
from smartloop.types import ModelCapability

logger = logging.getLogger(__name__)


class SkillProcessor:
    """Matches skills to a request and synthesizes their results."""

    # ``key=value`` pairs in a skill's prose, read as default tool arguments.
    _KV_PAIR_RE = re.compile(r'(\w+)\s*=\s*(\w+)')

    # NOTE: find_server_hint, extract_default_args and compact_instruction
    # have no callers — argument resolution goes through MCPArgumentExtractor
    # and presentation guidance uses the skill body verbatim. Moved here as-is
    # rather than deleted during the split; safe to drop.

    def __init__(self, models: ModelRouter, explorer: Explorer):
        self._models = models
        self.explorer = explorer
        # Skills matched for the current request, kept so a later step can ask
        # what matched without re-running the match.
        self._matched_skills: Optional[List[Dict[str, str]]] = None

    def match(self, ctx: _OrchestratorContext, node: SkillMatchNode) -> NodeResult:
        """Select the user-picked skill, or LLM-match when none was picked."""
        self._models.emit(
            "analyze", "processing",
            f"Analyzing your request against {len(ctx.skills)} available skill(s)",
        )
        parsed_skills: List[Dict[str, str]] = []
        for sc in ctx.skills:
            name, desc = "", ""
            for line in sc.split("\n"):
                s = line.strip()
                if s.startswith("name:"):
                    name = s[5:].strip().strip("'\"")
                elif s.startswith("description:"):
                    desc = s[12:].strip().strip("'\"")

            if name:
                skill_map = self._parse_skill_map(sc)
                parsed_skills.append({
                    "name": name, "description": desc,
                    "content": sc,
                    "skill_map": skill_map,
                })
        ctx.parsed_skills = parsed_skills

        _SYSTEM_TOOL_NAMES = {
            FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL, BASH_TOOL, EXPLORE_TOOL,
        }

        # The user explicitly picked a skill for this turn (the API resolved a
        # leading ``/fund-summary`` to a skill name). That IS the intent signal,
        # so use it directly and skip the LLM detection — a second LLM guess must
        # not be able to veto an explicit choice. Only a request naming a built-in
        # tool skill or a ``__``-prefixed system skill (never user-selectable), or
        # a name that matches nothing, falls through to normal detection.
        if ctx.requested_skill:
            def _norm(s: str) -> str:
                return re.sub(r'[\s_\-]+', '', s.lower())
            wanted = _norm(ctx.requested_skill)
            requested = [
                s for s in parsed_skills
                if _norm(s.get("name", "")) == wanted
                and s.get("name", "") not in _SYSTEM_TOOL_NAMES
                and not s.get("name", "").startswith("__")
            ]
            if requested:
                self._models.emit(
                    "analyze", "processing",
                    f"Fulfilling your request with the '{requested[0]['name']}' skill",
                )
                self._matched_skills = requested
                return NodeResult(
                    node_id=node.node_id, success=True,
                    data={"matched_skills": requested},
                )
            logger.info(
                "[skill_match] selected skill %r not selectable/not found among "
                "%s — using normal detection",
                ctx.requested_skill, [s.get("name", "") for s in parsed_skills],
            )

        _has_url = bool(re.search(r'https?://[^\s<>\'")\]]+', ctx.user_message))
        skills_for_matching = [
            s for s in parsed_skills
            if s.get("name", "") not in _SYSTEM_TOOL_NAMES
        ]
        if not _has_url:
            skills_for_matching = [
                s for s in skills_for_matching
                if not any(kw in s.get("name", "").lower() for kw in ("fetch", "browse", "scrape"))
            ]

        # Nothing to match — skip the LLM entirely so a project with no skills
        # never pays to load the heavier skills model.
        if not skills_for_matching:
            self._matched_skills = []
            return NodeResult(node_id=node.node_id, success=True, data={"matched_skills": []})

        # Skill matching is an intent judgement the small orchestrator model is
        # poor at (it defaults to picking the only listed skill). Prefer the
        # 'skills'-capability model, but only when it is resident or fits — a
        # memory-tight device falls back to the mini instead of risking an OOM.
        match_llm = self._models.decision_llm(ModelCapability.SKILLS)
        logger.info("[models] SKILL_MATCH → %s", self._models.model_label(match_llm))

        matched_skills = self.explorer.match_skills(
            ctx.user_message, skills_for_matching, llm=match_llm,
        )
        self._matched_skills = matched_skills
        return NodeResult(node_id=node.node_id, success=True, data={"matched_skills": matched_skills})

    def extract_tool_condition(
        self,
        skill_text: str,
        tool_name: str,
    ) -> Optional[str]:
        """Extract a conditional clause for a tool from skill text.

        Searches for the tool name in ``skill_text`` and uses the LLM
        to determine whether execution is conditional.  Returns an
        ``llm:`` condition string or ``None``.
        """
        tool_pos = skill_text.lower().find(tool_name.lower())
        if tool_pos < 0:
            tool_pos = skill_text.lower().find(
                tool_name.replace("_", " ").lower()
            )
        if tool_pos < 0:
            return None

        start = max(0, tool_pos - 300)
        end = min(len(skill_text), tool_pos + 200)
        relevant_content = skill_text[start:end]

        prompt = (
            "Read this skill instruction and determine if executing "
            f"the tool '{tool_name}' is conditional on some prior result.\n\n"
            f"Skill text:\n{relevant_content}\n\n"
            "If the tool should ONLY run when a condition is met, "
            "reply with a single YES/NO question that captures the "
            "condition (e.g. 'Were discrepancies found in the results?').\n"
            "If the tool should ALWAYS run (no condition), reply: NONE"
        )

        try:
            response = str(self._models.agent_llm.invoke(
                prompt, max_tokens=DECISION_MAX_TOKENS,
            )).strip()
            logger.info(f"[condition-extract] LLM response: {response[:120]}")

            if not response or "none" in response.lower()[:20]:
                return None

            question = response.strip().strip('"\'')
            # A valid condition is a YES/NO question. A bare "YES"/"NO" (or any
            # answer with no question mark) is a malformed reply, not a real
            # condition — treat it as unconditional rather than fabricating an
            # ``llm:YES`` gate that can never be satisfied.
            if "?" not in question:
                return None

            parts = question.split("?")
            for i in range(len(parts) - 1, -1, -1):
                candidate = parts[i].strip()
                if candidate and len(candidate) > 10:
                    question = candidate + "?"
                    break

            return f"llm:{question}"

        except Exception as e:
            logger.warning(f"[condition-extract] LLM failed: {e}")
            return None

    @staticmethod
    def find_server_hint(
        matched_skills: List[Dict[str, str]],
        tool_name: str,
    ) -> Optional[str]:
        """Find a server name hint near a tool reference in skill content.

        Looks for patterns like "save_issue on Linear MCP" or
        "tool in Linear MCP".
        """
        for sk in matched_skills:
            content = sk.get("content", "")
            # Pattern: <tool_name> on <Server>
            pattern = re.compile(
                rf'{re.escape(tool_name)}\s+on\s+(\w+)',
                re.IGNORECASE,
            )
            m = pattern.search(content)
            if m:
                return m.group(1)

            # Pattern: tool in <Server> MCP
            pattern2 = re.compile(
                r'tool\s+in\s+(\w+)\s+MCP',
                re.IGNORECASE,
            )
            m2 = pattern2.search(content)
            if m2:
                return m2.group(1)

        return None

    # Matches key=value pairs in skill prose, e.g. "team=DEV, assignee=me"
    @staticmethod
    def extract_default_args(
        matched_skills: List[Dict[str, str]],
        tool_name: str,
    ) -> Dict[str, str]:
        """Extract key=value default args from skill text near a tool reference.

        Scans the skill body for all occurrences of the tool name and
        picks up ``key=value`` pairs in the surrounding context.
        For example, given:

            "call save_issue on Linear MCP, use team=DEV, assignee=me"

        Returns ``{"team": "DEV", "assignee": "me"}``.
        """
        defaults: Dict[str, str] = {}
        _skip_keys = {"name", "description", "tools", "type", "step", "step_type"}
        for sk in matched_skills:
            content = sk.get("content", "")
            content_lower = content.lower()
            tool_lower = tool_name.lower()
            # Scan all occurrences of the tool name
            search_start = 0
            while True:
                tool_pos = content_lower.find(tool_lower, search_start)
                if tool_pos < 0:
                    break
                search_start = tool_pos + len(tool_lower)
                # Look at the surrounding window (200 chars before/after)
                start = max(0, tool_pos - 200)
                end = min(len(content), tool_pos + len(tool_name) + 200)
                window = content[start:end]
                for m in self._KV_PAIR_RE.finditer(window):
                    key, value = m.group(1), m.group(2)
                    if key.lower() in _skip_keys:
                        continue
                    defaults[key] = value
        if defaults:
            logger.info(
                f"[fallback] extracted default_args for '{tool_name}': {defaults}"
            )
        return defaults

    @staticmethod
    def strip_frontmatter(content: str) -> str:
        """Strip YAML frontmatter and ```mcp blocks from skill content.

        Returns the body of the skill — the instructions that should be
        injected into context for the streaming LLM.  ``\u0060\u0060\u0060mcp``
        blocks are execution directives, not prose — they must be
        removed so the LLM doesn't echo them verbatim.
        """
        lines = content.split("\n")
        body_start = 0
        in_frontmatter = False
        _FM_KEYS = {"name:", "description:", "tools:"}

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "---":
                if not in_frontmatter:
                    in_frontmatter = True
                    continue
                else:
                    # End of fenced frontmatter block
                    body_start = i + 1
                    break
            if in_frontmatter:
                continue
            # Unfenced frontmatter: lines starting with known keys
            if any(stripped.startswith(k) for k in _FM_KEYS):
                body_start = i + 1
                continue
            # First non-frontmatter, non-empty line
            if stripped:
                body_start = i
                break

        body = "\n".join(lines[body_start:]).strip()
        body = body if body else content

        # Defensively strip any legacy ```mcp / ```tool_code fenced blocks so
        # they never reach the streaming LLM as instructions.
        body = TOOL_CODE_FENCE_RE.sub("", body).strip()

        return body

    def synthesize(
        self,
        user_message: str,
        accumulated_context: str,
        skill_context: str,
    ) -> str:
        """Synthesize all accumulated step outputs into a single document.

        Falls back to the raw accumulated context if synthesis fails.
        """
        prompt = (
            "<system>\n"
            "You are a data synthesis assistant. You have been given raw outputs "
            "from several execution steps (tool calls, document retrieval, analysis). "
            "Your task is to merge them into a single, coherent, factual document "
            "that fully answers the user's request.\n\n"
            "Rules:\n"
            "- Include ALL factual values exactly as provided — do not round, "
            "estimate, or omit numbers.\n"
            "- Do NOT answer the user directly or add conversational framing.\n"
            "- Do NOT include JSON, code blocks, or raw data structures.\n"
            "- Write in clear prose with labelled sections if appropriate.\n"
            "- If there is only one data source, summarise it concisely.\n"
            "</system>\n"
            "<user>\n"
            f"User request: {user_message}\n\n"
            f"Skill context:\n{skill_context[:1000]}\n\n"
            f"Accumulated step outputs:\n{accumulated_context}\n"
            "</user>"
        )

        try:
            # Synthesis writes the final skill answer, so use the ranked large
            # model for the skills capability (largest-first via ranked_models_for),
            # not the small orchestrator. Falls back to the large/agent model when
            # no skills-capable model fits.
            synth_llm = self._models.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)
            logger.info(
                "[models] SYNTHESIZE → %s",
                self._models.model_label(synth_llm),
            )
            result = str(synth_llm.invoke(prompt)).strip()
            if result:
                logger.info(
                    f"[synthesize] produced {len(result)} chars "
                    f"from {len(accumulated_context)} chars of accumulated data"
                )
                return result
        except Exception as e:
            logger.warning(f"[synthesize] synthesis failed, using raw context: {e}")

        return accumulated_context

    def compact_instruction(
        self,
        skill_context: Optional[str],
        user_message: str,
    ) -> Optional[str]:
        """Distil a compact instruction from skill context for the streaming LLM.

        Extracts only presentation/formatting guidance.
        """
        if not skill_context or not skill_context.strip():
            return None

        prompt = (
            "<system>\n"
            "You are reading a skill instruction file. All tool calls and "
            "data-gathering steps have ALREADY been executed successfully.\n\n"
            "Extract ONLY guidance about how to PRESENT or FORMAT the "
            "results for the user — but ONLY if the skill text explicitly "
            "specifies a format. Do not invent formatting guidance.\n\n"
            "Do NOT include any tool call instructions, API actions, "
            "function calls, or internal execution directives — those are done.\n\n"
            "If the skill text does not explicitly specify a presentation "
            "format, reply with: NONE\n"
            "Otherwise reply with a concise 1-3 sentence instruction "
            "that quotes the presentation directive from the skill text.\n"
            "</system>\n"
            "<user>\n"
            f"User request: {user_message}\n\n"
            f"Skill text:\n{skill_context[:2000]}\n"
            "</user>"
        )

        try:
            result = str(self._models.agent_llm.invoke(
                prompt, max_tokens=SHORT_TEXT_MAX_TOKENS,
            )).strip()
            if not result or "none" in result.lower()[:10]:
                return None
            logger.info(f"[skill] compact instruction: {result[:200]}")
            return result
        except Exception as e:
            logger.warning(f"[skill] compact instruction extraction failed: {e}")
            # No fallback to raw skill body — it contains execution
            # directives that the streaming LLM would follow verbatim,
            # fabricating responses based on skill instructions rather
            # than actual data.
            return None

    @staticmethod
    def _parse_skill_map(content: str) -> List[Dict[str, str]]:
        """Parse a ``## skill_map`` section from skill markdown into step dicts.

        Each list item is a YAML-like block with keys: step, tool, server,
        instruction, step_type, condition.  Returns an empty list when no
        ``## skill_map`` section is found.
        """
        # Find the ## skill_map section
        match = re.search(r'(?m)^##\s+skill_map\s*\n(.*)', content, re.DOTALL)
        if not match:
            return []

        block = match.group(1)
        # Stop at the next markdown heading (if any)
        next_heading = re.search(r'(?m)^##\s', block)
        if next_heading:
            block = block[:next_heading.start()]

        steps: List[Dict[str, str]] = []
        current: Dict[str, str] = {}

        for line in block.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # New item starts with "- key: value"
            item_match = re.match(r'^-\s+(\w+):\s*(.*)', stripped)
            if item_match:
                # If this is the "step" key, save previous item and start new
                key, value = item_match.group(1), item_match.group(2).strip().strip('"\'')
                if key == "step" and current:
                    steps.append(current)
                    current = {}
                current[key] = value
            else:
                # Continuation key: value (no leading dash)
                kv_match = re.match(r'^(\w+):\s*(.*)', stripped)
                if kv_match:
                    key, value = kv_match.group(1), kv_match.group(2).strip().strip('"\'')
                    current[key] = value

        if current:
            steps.append(current)

        return steps[:5]