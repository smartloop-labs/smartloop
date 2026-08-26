"""Building the execution DAG for a request.

Every request runs the same shape: collect what is already indexed, classify
the intent, plan the steps, then wire one node per step into a graph that ends
at a single reason node.  Whether a skill matched changes what the planner is
told, not which path is taken — a matched skill contributes guiding context,
not its own control flow.

The graph is built lazily.  The route node cannot know what steps exist until
classification has run, so it returns its downstream nodes as ``_extend_dag``
and the executor grafts them on mid-walk.
"""
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from smartloop.agents.hybrid_executor import LLMExecutor
from smartloop.agents.nodes import (
    AgentNode, AgentNodeName, CollectNode, LLMDecisionNode, NodeResult,
    RouteNode, SearchNode, SkillMatchNode, ToolCallNode, TransformNode,
)
from smartloop.agents.fetch_agent import FetchAgent
from smartloop.agents.mcp.explorer import EnrichedContext, Explorer, SearchContext
from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
from smartloop.agents.orchestration.context import _OrchestratorContext
from smartloop.agents.orchestration.document_fanout import DocumentFanoutStep
from smartloop.agents.orchestration.judge_step import JudgeStep
from smartloop.agents.orchestration.model_router import ModelRouter
from smartloop.agents.orchestration.replies import is_placeholder_output
from smartloop.agents.orchestration.response_builder import ResponseBuilder
from smartloop.agents.orchestration.skill_processor import SkillProcessor
from smartloop.agents.orchestration.step_executor import (
    StepExecutor, plan_step_result,
)
from smartloop.agents.orchestration.tool_fanout import ToolFanout
from smartloop.agents.planner import (
    IntentType, ModelTier, Planner, PlanStep, StepType,
)
from smartloop.agents.web_search_agent import WebSearchAgent
from smartloop.types import ModelCapability

logger = logging.getLogger(__name__)

REASON_NODE = AgentNodeName.REASON


class PipelineBuilder:
    """Turns a request into the DAG of nodes that answers it."""

    def __init__(
        self,
        models: ModelRouter,
        planner: Planner,
        explorer: Explorer,
        executor: LLMExecutor,
        tool_caller: MCPToolCaller,
        fetch_agent: FetchAgent,
        web_search_agent: WebSearchAgent,
        judge: JudgeStep,
        doc_fanout: DocumentFanoutStep,
        skills: SkillProcessor,
        steps: StepExecutor,
        tool_fanout: ToolFanout,
        response: ResponseBuilder,
    ):
        self._models = models
        self.planner = planner
        self.explorer = explorer
        self._executor = executor
        self.tool_caller = tool_caller
        self._fetch_agent = fetch_agent
        self._web_search_agent = web_search_agent
        self._judge = judge
        self._doc_fanout = doc_fanout
        self._skills = skills
        self._steps = steps
        self._tool_fanout = tool_fanout
        self._response = response

    @property
    def agent_llm(self):
        """The decision model, owned by the router."""
        return self._models.agent_llm

    def handle_collect(
        self,
        ctx: _OrchestratorContext,
        deps: Dict,
        node: CollectNode,
        skill_match_node: SkillMatchNode,
    ) -> NodeResult:
        """Vector + workspace + feedback search."""
        logger.info("[models] COLLECT → no LLM (vector/feedback/workspace search)")
        matched_skills = deps[skill_match_node.node_id].data.get("matched_skills")
        enriched = ctx.enriched
        search_ctx = self.explorer.collect_parallel(
            ctx.user_message, ctx.model_service, ctx.effective_cwd,
            enriched=enriched,
            matched_skills=matched_skills if matched_skills else None,
            mcp_servers=ctx.mcp_servers,
            mcp_tool_contexts=ctx.mcp_tool_contexts,
            conversation_history=ctx.conversation_history,
            executor=self._executor._io_pool,
            project_dir=ctx.project_dir,
        )
        return NodeResult(node_id=node.node_id, success=True, data={"search_ctx": search_ctx})

    def handle_route(
        self,
        ctx: _OrchestratorContext,
        deps: Dict,
        node: RouteNode,
        skill_match_node: SkillMatchNode,
        collect_node: CollectNode,
    ) -> NodeResult:
        """Decision: build the unified pipeline. Dynamically adds downstream nodes."""
        logger.info("[models] ROUTE (classify+plan) → agent_llm: %s", self._models.model_label(self.agent_llm))
        matched_skills = deps[skill_match_node.node_id].data.get("matched_skills")
        search_ctx = deps[collect_node.node_id].data.get("search_ctx")
        enrich_ctx = ctx.enriched
        _skill_names = ", ".join(s.get("name", "") for s in (matched_skills or []) if s.get("name"))
        self._models.emit(
            "plan", "processing",
            f"Planning the response using skill(s): {_skill_names}"
            if _skill_names else "Planning the response and selecting tools",
        )

        if matched_skills:
            ctx.skill_bodies = [
                self._skills.strip_frontmatter(s["content"]) for s in matched_skills
            ]
            ctx.skill_context = "\n\n".join(ctx.skill_bodies)

        extend_dag: List[tuple] = []
        parent: AgentNode = node

        if self._doc_fanout.applies_to(ctx, search_ctx):
            contributed = self._doc_fanout.build(ctx, search_ctx, parent)
            extend_dag.extend(contributed.pairs)
            parent = contributed.parent

        extend_dag.extend(
            self._build_unified_pipeline(ctx, matched_skills, search_ctx, enrich_ctx, parent=parent)
        )

        return NodeResult(
            node_id=node.node_id, success=True,
            data={
                "search_ctx": search_ctx,
                "enrich_ctx": enrich_ctx,
                "_extend_dag": extend_dag,
            },
        )

    def _build_unified_pipeline(
        self,
        ctx: _OrchestratorContext,
        matched_skills: Optional[List[Dict[str, str]]],
        search_ctx: SearchContext,
        enrich_ctx: EnrichedContext,
        parent: AgentNode = None,
    ) -> List[tuple]:
        """Build a single pipeline regardless of whether skills matched.

        Both paths produce a uniform ``List[PlanStep]`` that feeds into
        the same node builders and handlers — no branching downstream.
        """
        plan_steps = self._derive_steps(ctx, search_ctx, enrich_ctx)
        if plan_steps:
            logger.info(
                "[plan] %d step(s): %s",
                len(plan_steps),
                " → ".join(
                    f"{s.step_type.value}:{s.tool_name}[{s.model_tier.value}]"
                    for s in plan_steps
                ),
            )

        return self.build_step_nodes(
            ctx, plan_steps, search_ctx, parent=parent,
        )

    def _derive_steps(
        self,
        ctx: _OrchestratorContext,
        search_ctx: SearchContext,
        enrich_ctx: EnrichedContext,
    ) -> List[PlanStep]:
        """Derive execution steps for any request through one pipeline.

        Whether or not the prompt matched a skill, the request runs the
        same classify → create_plan flow.  An LLM-matched skill is injected
        as planning context — its instructions guide step generation — so
        the planner produces and executes the skill's procedure, instead of
        taking a separate derivation path.
        """
        vector_context = search_ctx.vector_context

        # The LLM skill-match already decided which skills (if any) the prompt
        # maps to.  Feed each matched skill's stripped instructions to the
        # planner as a distinct entry so it can plan across multiple matched
        # skills and accumulate their outputs into the final result.  If
        # nothing matched, plan from the user prompt alone — do NOT dump every
        # workspace skill back in.
        planner_skills = ctx.skill_bodies or None

        # Planning is always driven by the user prompt's own classification.
        # A matched skill only contributes guiding context (planner_skills) —
        # it does not change the control flow.
        classification = self.planner.classify(
            ctx.user_message,
            mcp_tools=ctx.mcp_tool_contexts,
            skills=planner_skills,
            document_context=vector_context or ctx.document_context,
            has_attachments=ctx.has_attachments,
            history=enrich_ctx.history_insight,
            cwd=ctx.effective_cwd,
            available_capabilities=ctx.available_capabilities or None,
        )
        _pre_promotion_intent = classification.intent

        # Indexed context that can ground an EXPLORE answer.  All four
        # sources count equally — a workspace-file or feedback match is just
        # as real as a document match (each is only present when it cleared
        # its search relevance threshold).  Counting only documents/skill-tools
        # here caused workspace-only matches to be discarded and the query
        # answered by the small model from raw guesswork.
        matched_sources = []
        if vector_context:
            matched_sources.append("documents")
        if search_ctx.workspace_context:
            matched_sources.append("workspace")
        if search_ctx.feedback_context:
            matched_sources.append("feedback")
        if search_ctx.web_context:
            matched_sources.append("web")
        if search_ctx.mcp_context:
            matched_sources.append("mcp")
        if search_ctx.mcp_document_context:
            matched_sources.append("mcp-content")
        has_indexed_context = bool(matched_sources)

        if classification.intent == IntentType.CONVERSATION:
            if has_indexed_context:
                classification.intent = IntentType.EXPLORE
                classification.reasoning = (
                    f"indexed context available ({', '.join(matched_sources)}) — exploring"
                )
        elif classification.intent == IntentType.EXPLORE:
            # Small classifier LLM often hallucinates EXPLORE for plain
            # question queries.  If there's nothing indexed to explore,
            # treat it as conversation — otherwise we'd waste a search
            # step and route the answer through the small model.
            if not has_indexed_context:
                classification.intent = IntentType.CONVERSATION
                classification.reasoning = (
                    "no indexed context to explore — treating as conversation"
                )

        if classification.detected_urls and classification.intent == IntentType.CONVERSATION:
            classification.intent = IntentType.TOOL_CALL
            classification.reasoning = f"URL detected — deterministic fetch: {classification.detected_urls[0]}"

        logger.info(
            f"[classify] {classification.intent.value}: "
            f"{classification.reasoning}"
        )
        if classification.intent != _pre_promotion_intent:
            logger.info(
                "[classify] intent adjusted %s → %s (%s)",
                _pre_promotion_intent.value,
                classification.intent.value,
                classification.reasoning,
            )

        plan_context = self.planner.build_context(
            ctx.user_message, ctx.available_servers,
            cwd=ctx.effective_cwd,
            skills=planner_skills,
            document_context=vector_context or ctx.document_context,
        )
        # Multi-step tool selection is complex reasoning — prefer the large
        # 'dp' model when it is resident or fits, else fall back to the mini.
        # Resolved lazily, only for MULTI_STEP, so simple requests never touch
        # the 4B and a memory-tight device is never pushed into an OOM.
        plan_llm = None
        if classification.intent == IntentType.MULTI_STEP:
            plan_llm = self._models.decision_llm(ModelCapability.DP)
            logger.info("[models] PLAN (multi-step) → %s", self._models.model_label(plan_llm))

        plan = self.planner.create_plan(classification, plan_context, llm=plan_llm)
        if plan.steps:
            plan.steps = self.explorer.resolve_fetch_steps(
                plan.steps, ctx.available_servers, ctx.user_message
            )

        # Skills are indexed and matched by user intent; their plain-text body
        # guides the LLM directly. Tool execution is driven by the planner /
        # intent-matched MCP tools, not by parsing skill code blocks.
        has_skill = bool(ctx.skill_context)

        detected_capability = classification.capability
        for step in plan.steps:
            # Planner-emitted CONVERSATIONAL steps carry their own tier +
            # capability (LARGE / knowledge) — don't clobber them.
            if step.step_type == StepType.CONVERSATIONAL and step.capability:
                continue
            if has_skill and step.step_type == StepType.TOOL:
                step.model_tier = ModelTier.LARGE
                step.capability = detected_capability or ModelCapability.SKILLS
            else:
                step.model_tier = ModelTier.SMALL
                if detected_capability:
                    step.capability = detected_capability

        return plan.steps

    def build_step_nodes(
        self,
        ctx: _OrchestratorContext,
        plan_steps: List[PlanStep],
        search_ctx: SearchContext,
        parent: AgentNode = None,
    ) -> List[tuple]:
        """Build DAG nodes from plan steps.

        One code path regardless of whether steps came from a skill or
        from the classify-then-plan pipeline.  Handlers use
        ``ctx.skill_context`` (set by the route handler when skills
        matched) for enrichment — no branching on matched_skills.
        """
        is_conversation = not plan_steps

        # The judge runs on both routes.  An EXPLORE step reads the retrieval
        # sources straight off search_ctx, so judging them only on the
        # conversation route let unrelated workspace / feedback / MCP content
        # walk into the answer context whenever the planner emitted any step.
        nodes_and_handlers: List[tuple] = []
        if self._judge.applies_to(ctx, search_ctx):
            judged = self._judge.build(ctx, search_ctx, parent)
            nodes_and_handlers.extend(judged.pairs)
            parent = judged.parent

        if is_conversation:
            reason_node = TransformNode(
                node_id=REASON_NODE, depends_on=[parent],
            )
            reason_handler = self.make_reason_handler(ctx, search_ctx, is_conversation)
            nodes_and_handlers.append((reason_node, reason_handler))
            return nodes_and_handlers

        prev_nodes: List[AgentNode] = [parent]

        # Skill-derived steps run sequentially — they often have implicit
        # dependencies (e.g. conditions extracted from skill text at resolve
        # time) that _group_independent_steps cannot detect.
        if ctx.skill_context:
            groups = [[step] for step in plan_steps]
        else:
            groups = self._steps.group_independent_steps(plan_steps)

        pre_step_nodes: List[AgentNode] = []

        for gi, group in enumerate(groups):
            group_nodes: List[AgentNode] = []
            for step in group:
                if step.step_type == StepType.TOOL:
                    if step.for_each:
                        fanout_node = LLMDecisionNode(depends_on=list(prev_nodes))
                        fanout_handler = self._tool_fanout.make_handler(ctx, step, search_ctx)
                        nodes_and_handlers.append((fanout_node, fanout_handler))
                        group_nodes.append(fanout_node)
                        pre_step_nodes.append(fanout_node)
                    else:
                        resolve_node = LLMDecisionNode(depends_on=list(prev_nodes))
                        resolve_handler = self.make_resolve_handler(
                            ctx, step, search_ctx,
                            has_upstream=bool(pre_step_nodes),
                        )
                        nodes_and_handlers.append((resolve_node, resolve_handler))

                        execute_node = ToolCallNode(depends_on=[resolve_node])
                        execute_handler = self._make_execute_handler(ctx, step)
                        nodes_and_handlers.append((execute_node, execute_handler))

                        group_nodes.append(execute_node)
                        pre_step_nodes.append(execute_node)
                else:
                    if step.step_type == StepType.EXPLORE:
                        node = SearchNode(depends_on=list(prev_nodes))
                    elif step.step_type == StepType.ANALYZE:
                        node = LLMDecisionNode(depends_on=list(prev_nodes))
                    else:
                        node = ToolCallNode(depends_on=list(prev_nodes))

                    handler = self.make_step_handler(ctx, step, search_ctx)
                    nodes_and_handlers.append((node, handler))
                    group_nodes.append(node)
                    pre_step_nodes.append(node)

            prev_nodes = group_nodes

        reason_node = TransformNode(
            node_id=REASON_NODE, depends_on=pre_step_nodes,
        )
        reason_handler = self.make_reason_handler(ctx, search_ctx, is_conversation)
        nodes_and_handlers.append((reason_node, reason_handler))
        return nodes_and_handlers

    def make_step_handler(
        self,
        ctx: _OrchestratorContext,
        step: PlanStep,
        search_ctx: SearchContext,
    ) -> Callable:
        """Create a handler for a non-TOOL step node.

        Works uniformly for both skill-derived and prompt-derived steps.
        Uses ``ctx.skill_context`` for instruction enrichment when available.
        """
        def handler(deps: Dict) -> NodeResult:
            prev_results = self._response.collect_raw_from_deps(deps)
            node_id = f"step_{step.step}_{step.tool_name or step.step_type.value}"

            # FETCH: run the FetchAgent (HTTP GET → markdown) in-process
            if step.step_type == StepType.FETCH:
                url = (step.default_args or {}).get("url", "")
                if not url:
                    urls = re.findall(r'https?://[^\s<>\'")\]]+', step.instruction or "")
                    url = urls[0] if urls else ""
                if not url:
                    logger.warning("[fetch] No URL to fetch for step %s", step.step)
                    return plan_step_result(node_id, step, "")
                self._models.emit("fetch", "processing", f"Fetching {url}")
                content = self._fetch_agent.fetch(url)
                # Return empty on failure so downstream reasoning falls back to
                # any explore/vector context instead of surfacing a fetch error.
                if not content:
                    logger.info(f"[fetch] step {step.step}: empty result from {url}")
                    return plan_step_result(node_id, step, "")
                logger.info(f"[fetch] step {step.step}: {len(content)} chars from {url}")
                return plan_step_result(node_id, step, content, content)

            # WEB_SEARCH: run the WebSearchAgent (latest info/events) in-process
            if step.step_type == StepType.WEB_SEARCH:
                query = (step.default_args or {}).get("query") or step.instruction or ctx.user_message
                self._models.emit("web_search", "processing", "Searching the web for the latest information")
                content = self._web_search_agent.search(query)
                # Record the URLs on the shared search context, exactly as the
                # explore path does (see Explorer.collect).  _build_final_result
                # reads them back as citations, so without this a plan that
                # reaches the web through a WEB_SEARCH step answers from live
                # pages while surfacing no reference links at all.
                extracts = getattr(
                    self._web_search_agent, "last_source_extracts", None
                ) or {}
                for url in self._web_search_agent.last_sources:
                    if url not in search_ctx.web_sources:
                        search_ctx.web_sources.append(url)
                    if url in extracts:
                        search_ctx.web_source_extracts[url] = extracts[url]
                logger.info(
                    "[web_search] step %s: %d chars, %d source(s)",
                    step.step, len(content or ""),
                    len(self._web_search_agent.last_sources),
                )
                return plan_step_result(node_id, step, content or "", content or "")

            # EXPLORE: return pre-fetched search context (already collected)
            if step.step_type == StepType.EXPLORE:
                # Local sources first, the web last. Everything above the web
                # is the user's own material — their documents, workspace,
                # connected servers and prior feedback — and a model reading
                # top-down should meet that before a public page. Web results
                # used to sit third, ahead of both MCP sources, which put
                # someone else's page in front of the user's own data.
                parts = []
                if search_ctx.vector_context:
                    parts.append(search_ctx.vector_context)
                if search_ctx.workspace_context:
                    parts.append(search_ctx.workspace_context)
                if search_ctx.mcp_context:
                    parts.append(search_ctx.mcp_context)
                if search_ctx.mcp_document_context:
                    parts.append(
                        f"MCP CONTENT:\n{search_ctx.mcp_document_context}"
                    )
                if search_ctx.feedback_context:
                    parts.append(search_ctx.feedback_context)
                if search_ctx.web_context:
                    parts.append(self._response.frame_web_results(search_ctx.web_context))
                content = "\n\n---\n\n".join(parts) if parts else ""
                if search_ctx.citations:
                    ctx.explore_citations = search_ctx.citations
                return plan_step_result(node_id, step, content)

            # ANALYZE: assess relevance of explore output via LLM
            if step.step_type == StepType.ANALYZE:
                explore_content = ""
                for dep_result in deps.values():
                    if dep_result.success and isinstance(dep_result.data, dict):
                        raw = dep_result.data.get("raw", "")
                        if raw:
                            explore_content += raw + "\n\n"
                explore_content = explore_content.strip()

                if not explore_content:
                    return plan_step_result(node_id, step, "")

                analyze_prompt = f"""\
<system>
You are a relevance assessment assistant.
Given a user request and search results, determine if the search results contain information relevant to the request.
If relevant, produce a concise analysis of the useful information.
If NOT relevant, respond with exactly: NOT_RELEVANT
</system>
<user>
User request: {ctx.user_message}

Search results:
{explore_content[:6000]}

Assess whether these search results are relevant to the user's request. If relevant, summarize the key findings. If not relevant, respond with NOT_RELEVANT.
</user>"""

                try:
                    step_llm = self._models.llm_for_tier(step.model_tier, step.capability)
                    logger.info(
                        "[models] ANALYZE(%s) → %s%s: %s",
                        step.step, step.model_tier.value,
                        f"/{step.capability}" if step.capability else "",
                        self._models.model_label(step_llm),
                    )
                    result = str(step_llm.invoke(analyze_prompt)).strip()
                    if (result == "NOT_RELEVANT" or len(result) < 20
                            or is_placeholder_output(result)):
                        logger.info("[analyze] context deemed not relevant — passing through raw")
                        return plan_step_result(node_id, step, explore_content, explore_content)
                    logger.info(f"[analyze] produced {len(result)} chars")
                    ctx.skill_report = result
                    return plan_step_result(node_id, step, result, result)
                except Exception as e:
                    logger.warning(f"[analyze] LLM analysis failed: {e}")
                    return plan_step_result(node_id, step, explore_content, explore_content)

            # Evaluate condition
            if step.condition:
                if not self._steps.evaluate_condition(step.condition, prev_results):
                    logger.info(f"[step] {step.step} SKIPPED — condition not met: {step.condition}")
                    return plan_step_result(node_id, step, "")

            # Enrich instruction with skill context when available
            exec_step = step
            if ctx.skill_context:
                enriched_instruction = step.instruction or ""
                if ctx.skill_context:
                    enriched_instruction = f"{enriched_instruction}\n\n{ctx.skill_context}" if enriched_instruction else ctx.skill_context
                exec_step = PlanStep(
                    step=step.step,
                    step_type=step.step_type,
                    tool_name=step.tool_name,
                    server_id=step.server_id,
                    server_name=step.server_name,
                    instruction=enriched_instruction,
                    condition=step.condition,
                )

            exec_llm = self._models.llm_for_tier(step.model_tier, step.capability)
            logger.info(
                "[models] EXEC(%s/%s) → %s%s: %s",
                step.step, step.step_type.value,
                step.model_tier.value,
                f"/{step.capability}" if step.capability else "",
                self._models.model_label(exec_llm),
            )

            result = self._steps.execute(
                ctx,
                step=exec_step,
                previous_results=prev_results if prev_results else None,
                # Prefer the reduced document context (the DP model's extraction,
                # written to search_ctx.vector_context by the doc-reduce node) so a
                # skill step running on a *different* model still sees it — matching
                # the plan-build path. Fall back to the original doc context.
                document_context=search_ctx.vector_context or ctx.document_context,
                llm=exec_llm,
            )

            if result:
                formatted, raw = result
                if step.step_type == StepType.ANALYZE and raw:
                    ctx.skill_report = raw
                return plan_step_result(node_id, step, formatted, raw)
            return plan_step_result(node_id, step, "")

        return handler

    def make_resolve_handler(
        self,
        ctx: _OrchestratorContext,
        step: PlanStep,
        search_ctx: SearchContext,
        has_upstream: bool = True,
    ) -> Callable:
        """Create a resolve handler for a TOOL step (parameter resolution).

        Works uniformly for both skill-derived and prompt-derived steps.
        Uses ``ctx.skill_context`` for condition extraction when available.

        ``has_upstream`` is False for the first tool step in the plan. A
        condition gates a tool on a *prior* result, so it is meaningless for
        a step with nothing upstream — evaluating one there would always skip
        (no prior result can satisfy it). This is exactly what happens for a
        lone fetch step whose skill text reads "when the user requests a URL,
        fetch it": the extractor misreads that applicability clause as a
        runtime condition and drops the only step. Skip condition gating when
        there is no upstream.
        """
        step_llm = self._models.llm_for_tier(step.model_tier, step.capability)
        logger.info(
            "[models] RESOLVE(%s) → %s%s: %s",
            step.tool_name, step.model_tier.value,
            f"/{step.capability}" if step.capability else "",
            self._models.model_label(step_llm),
        )
        def handler(deps: Dict) -> NodeResult:
            prev_results = self._response.collect_raw_from_deps(deps)
            node_id = f"step_{step.step}_{step.tool_name or step.step_type.value}_resolve"

            condition = step.condition if has_upstream else None
            if has_upstream and not condition and ctx.skill_context:
                condition = self._skills.extract_tool_condition(
                    ctx.skill_context, step.tool_name or "",
                )
            if condition:
                if not prev_results:
                    logger.info(f"[resolve] SKIPPED {step.tool_name} — no upstream context for condition")
                    return NodeResult(node_id=node_id, success=True, data={
                        "skip": True,
                        "tool_name": step.tool_name or step.step_type.value,
                    })
                if not self._steps.evaluate_condition(condition, prev_results):
                    logger.info(f"[resolve] SKIPPED {step.tool_name} — condition not met")
                    return NodeResult(node_id=node_id, success=True, data={
                        "skip": True,
                        "tool_name": step.tool_name or step.step_type.value,
                    })

            logger.info(
                f"[resolve] {step.step} [{step.step_type.value}] "
                f"{step.server_name}/{step.tool_name}: {step.instruction}"
            )

            resolved = self._steps.resolve_tool_args(
                ctx,
                step=step,
                previous_results=prev_results if prev_results else None,
                llm=step_llm,
            )

            if resolved is None:
                logger.info(f"[resolve] {step.tool_name} — could not resolve args")
                return NodeResult(node_id=node_id, success=True, data={
                    "skip": True,
                    "tool_name": step.tool_name or step.step_type.value,
                })

            server, tool_name, arguments = resolved
            logger.info(f"[resolve] {tool_name} resolved args: {arguments}")
            return NodeResult(node_id=node_id, success=True, data={
                "skip": False,
                "tool_name": tool_name,
                "server_id": server.id,
                "arguments": arguments,
                "tool_schema": self.tool_caller.find_tool_schema(server, tool_name),
            })

        return handler

    def _make_execute_handler(
        self,
        ctx: _OrchestratorContext,
        step: PlanStep,
    ) -> Callable:
        """Create an execute handler for a TOOL step (tool invocation).

        Reads resolved arguments from the resolve node and calls the tool.
        No LLM needed — just an API call.
        """
        logger.info("[models] EXECUTE(%s) → no LLM (MCP tool call)", step.tool_name)
        def handler(deps: Dict) -> NodeResult:
            node_id = f"step_{step.step}_{step.tool_name or step.step_type.value}_execute"

            # Get the resolve node's output
            resolve_data = None
            for dep_id, dep_result in deps.items():
                if dep_result.success and isinstance(dep_result.data, dict):
                    resolve_data = dep_result.data
                    break

            if not resolve_data or resolve_data.get("skip"):
                tool_name = resolve_data.get("tool_name", step.tool_name or step.step_type.value) if resolve_data else (step.tool_name or step.step_type.value)
                fail_msg = f"Tool '{tool_name}' skipped — could not resolve arguments." if step.skill_name else ""
                return plan_step_result(node_id, step, fail_msg)

            tool_name = resolve_data["tool_name"]
            server_id = resolve_data["server_id"]
            arguments = resolve_data["arguments"]

            # Find the server object
            server = self.tool_caller.find_server(ctx.mcp_servers, server_id, server_id)
            if not server:
                logger.warning(f"[execute] Server not found for {tool_name} (id={server_id})")
                fail_msg = f"Tool '{tool_name}' failed: server not found." if step.skill_name else ""
                return plan_step_result(node_id, step, fail_msg)

            result = self._steps.call_tool(server, tool_name, arguments, cwd=ctx.effective_cwd)
            if result is None:
                fail_msg = f"Tool '{tool_name}' call failed — no result returned." if step.skill_name else ""
                return plan_step_result(node_id, step, fail_msg)

            formatted, raw = result
            return plan_step_result(node_id, step, formatted, raw)

        return handler

    _FANOUT_MAX_ITEMS = 20

    def make_reason_handler(
        self,
        ctx: _OrchestratorContext,
        search_ctx: SearchContext,
        is_conversation: bool,
    ) -> Callable:
        """Create the reason handler — the final DAG node.

        Collects all upstream step outputs.  When skill context is
        available, synthesizes them into one coherent document.
        """
        logger.info("[models] REASON → agent_llm: %s", self._models.model_label(self.agent_llm))
        def handler(deps: Dict) -> NodeResult:
            all_accumulated: List[Dict[str, Any]] = []
            tool_results: List[Dict[str, Any]] = []

            # Every step output is judged against the user's intent before it
            # is appended.  A plan routinely runs steps that turn out not to
            # bear on the question, and each one that lands here becomes part
            # of the context the answer is told to work from.
            rejected: List[Dict[str, Any]] = []
            for dep_id, dep_result in deps.items():
                if not dep_result.success or not isinstance(dep_result.data, dict):
                    continue
                content = dep_result.data.get("content", "")
                raw = dep_result.data.get("raw", "")
                tool_name = dep_result.data.get("tool_name", "")
                is_tool = dep_result.data.get("is_tool_result", False)
                text = self._response.strip_tool_code_blocks(content or raw)
                if not text:
                    continue
                entry = {"tool_name": tool_name, "content": text, "_is_tool": is_tool}
                if ctx.skill_context or self._judge.serves_intent(
                    ctx.user_message, tool_name or "step output", text,
                ):
                    all_accumulated.append(entry)
                else:
                    rejected.append(entry)

            # If the judge rejected every step output, keep them all.  Losing
            # one output among several is the judge doing its job; losing the
            # lot after the plan ran is far more likely to be the judge itself
            # misfiring, and the request would answer from nothing at all.
            # (Retrieval that is entirely irrelevant is dropped earlier, at the
            # gate — there the empty result is the correct answer and falls
            # through to a plain conversational reply.)
            if rejected and not all_accumulated:
                logger.info(
                    "[judge] all %d step output(s) rejected — keeping them "
                    "rather than answering from nothing",
                    len(rejected),
                )
                all_accumulated = rejected

            for entry in all_accumulated:
                if entry.pop("_is_tool", False):
                    tool_results.append(entry)
            citations = ctx.explore_citations

            synthesized = None
            if ctx.skill_context and all_accumulated:
                parts = [r["content"] for r in all_accumulated if r["content"]]
                # Include the reduced document context (the DP model's extraction)
                # in the synthesis input so a skill run on a *different* model
                # still grounds its answer in the processed documents. Skip when a
                # step already carried it forward to avoid duplicating the text.
                doc_ctx = search_ctx.vector_context
                if doc_ctx and not any(doc_ctx in p for p in parts):
                    parts.insert(0, doc_ctx)
                if parts:
                    self._models.emit(
                        "synthesize", "processing",
                        f"Synthesizing the final answer from {len(parts)} result(s)",
                    )
                    raw_combined = "\n\n---\n\n".join(parts)
                    synthesized = self._skills.synthesize(
                        user_message=ctx.user_message,
                        accumulated_context=raw_combined,
                        skill_context=ctx.skill_context,
                    )

            return NodeResult(node_id=REASON_NODE, success=True, data={
                "accumulated": all_accumulated,
                "tool_results": tool_results,
                "synthesized": synthesized,
                "search_ctx": search_ctx,
                "is_conversation": is_conversation,
                "citations": citations,
            })

        return handler