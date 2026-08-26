"""SLPOrchestrator — the coordinator for a user request.

Three phases, and little else: build the request context, walk the DAG the
pipeline builder assembles, then hand the accumulated results to the response
builder.  Everything the phases actually do lives in
:mod:`smartloop.agents.orchestration`, one concern per module.

The DAG is what makes the phases possible: each stage (skill match, collect,
route, per-step, reason) is a node with explicit dependencies, so the executor
runs independent branches in parallel and sequential ones in order, without
this class knowing the shape in advance.
"""
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from smartloop.llms import BaseLLM
from smartloop.workspace.workspace import MCPServer
from smartloop.agents.hybrid_executor import LLMExecutor
from smartloop.agents.dag_executor import DAGExecutor
from smartloop.agents.nodes import (
    DAG, AgentNodeName, CollectNode, NodeResult, RouteNode, SkillMatchNode,
)
from smartloop.agents.planner import MCPToolContext, Planner
from smartloop.agents.mcp.server_selector import EXPLORE_TOOL
from smartloop.agents.mcp.mcp_agent import MCPAgent
from smartloop.agents.mcp.argument_extractor import MCPArgumentExtractor
from smartloop.agents.mcp.dependency_analyzer import MCPDependencyAnalyzer
from smartloop.agents.mcp.result_mapper import MCPResultMapper
from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
from smartloop.agents.mcp.explorer import Explorer
from smartloop.agents.fetch_agent import FetchAgent
from smartloop.agents.web_search_agent import WebSearchAgent
from smartloop.utils.mcp_result_formatter import ResultFormatter

from smartloop.agents.orchestration.context import (
    _OrchestratorContext, MCPExecutionResult,
)
from smartloop.agents.orchestration.model_router import ModelRouter
from smartloop.agents.orchestration.judge_step import JudgeStep
from smartloop.agents.orchestration.document_fanout import DocumentFanoutStep
from smartloop.agents.orchestration.skill_processor import SkillProcessor
from smartloop.agents.orchestration.step_executor import StepExecutor
from smartloop.agents.orchestration.tool_fanout import ToolFanout
from smartloop.agents.orchestration.response_builder import ResponseBuilder
from smartloop.agents.orchestration.pipeline_builder import PipelineBuilder

logger = logging.getLogger(__name__)

REASON_NODE = AgentNodeName.REASON

class SLPOrchestrator:
    """Coordinates plan → run → reason for one user message.

    Owns the collaborators and the orchestration components, and nothing else:
    every phase below delegates.  ``orchestrator_llm`` and ``agent_llm`` are
    exposed as write-through properties over :class:`ModelRouter`, which is
    where the models actually live.
    """

    def __init__(
        self,
        orchestrator_llm: BaseLLM,
        agent_llm: Optional[BaseLLM] = None,
        large_llm: Optional[BaseLLM] = None,
        large_llm_provider: Optional[Callable[[], Optional[BaseLLM]]] = None,
        capability_llm_provider: Optional[Callable[[str], Optional[BaseLLM]]] = None,
        capability_available: Optional[Callable[[str], bool]] = None,
        capability_skip_reason: Optional[Callable[[], Optional[tuple]]] = None,
        model_path: Optional[str] = None,
        project_id: Optional[str] = None,
        executor: Optional[LLMExecutor] = None,
    ):
        # ModelRouter owns the models; ``orchestrator_llm`` / ``agent_llm``
        # below read through to it so there is exactly one place they live.
        self._models = ModelRouter(
            orchestrator_llm=orchestrator_llm,
            agent_llm=agent_llm or orchestrator_llm,
            large_llm=large_llm,
            large_llm_provider=large_llm_provider,
            capability_llm_provider=capability_llm_provider,
            capability_available=capability_available,
            capability_skip_reason=capability_skip_reason,
        )

        # -- Collaborators shared by the components below --
        self.planner = Planner(self.agent_llm)
        self.arg_extractor = MCPArgumentExtractor(self.agent_llm)
        self.dep_analyzer = MCPDependencyAnalyzer()
        self.result_mapper = MCPResultMapper()
        self.tool_caller = MCPToolCaller(self.arg_extractor, self.result_mapper)
        self.result_formatter = ResultFormatter(orchestrator_llm)
        self.explorer = Explorer(self.agent_llm, self.tool_caller, self.arg_extractor)
        fetch_agent = FetchAgent()
        web_search_agent = WebSearchAgent(llm=self.agent_llm)
        # MCPAgent: vector-retrieves candidate tools as one flat server.tool_name
        # list, then LLM-picks. This is the resolution path for external MCP tool
        # steps (no separate server selector, no keyword-scan fallback).
        self._mcp_agent = MCPAgent(self.agent_llm, logger)

        # -- Orchestration components, constructed in dependency order --
        self._response = ResponseBuilder(orchestrator_llm)
        self._judge = JudgeStep(self._models)
        self._doc_fanout = DocumentFanoutStep(self._models)
        self._skills = SkillProcessor(self._models, self.explorer)
        self._steps = StepExecutor(
            orchestrator_llm=orchestrator_llm,
            tool_caller=self.tool_caller,
            arg_extractor=self.arg_extractor,
            dep_analyzer=self.dep_analyzer,
            result_formatter=self.result_formatter,
            explorer=self.explorer,
            mcp_agent=self._mcp_agent,
            response=self._response,
        )
        self._tool_fanout = ToolFanout(
            models=self._models,
            steps=self._steps,
            response=self._response,
            tool_caller=self.tool_caller,
        )

        # Prefer a shared, long-lived executor.  Fall back to a per-instance
        # one when none is supplied (e.g. in tests).
        if executor is not None:
            self._executor = executor
        else:
            self._executor = LLMExecutor(max_io_workers=6)
        self._dag_executor = DAGExecutor(max_workers=6)

        self._pipeline = PipelineBuilder(
            models=self._models,
            planner=self.planner,
            explorer=self.explorer,
            executor=self._executor,
            tool_caller=self.tool_caller,
            fetch_agent=fetch_agent,
            web_search_agent=web_search_agent,
            judge=self._judge,
            doc_fanout=self._doc_fanout,
            skills=self._skills,
            steps=self._steps,
            tool_fanout=self._tool_fanout,
            response=self._response,
        )

    @property
    def _fetch_agent(self) -> FetchAgent:
        """The HTTP fetcher.  Owned by :class:`PipelineBuilder`."""
        return self._pipeline._fetch_agent

    @_fetch_agent.setter
    def _fetch_agent(self, agent: FetchAgent) -> None:
        self._pipeline._fetch_agent = agent

    @property
    def _web_search_agent(self) -> WebSearchAgent:
        """The web searcher.  Owned by :class:`PipelineBuilder`."""
        return self._pipeline._web_search_agent

    @_web_search_agent.setter
    def _web_search_agent(self, agent: WebSearchAgent) -> None:
        self._pipeline._web_search_agent = agent

    @property
    def orchestrator_llm(self) -> BaseLLM:
        """The response model.  Owned by :class:`ModelRouter`."""
        return self._models.orchestrator_llm

    @orchestrator_llm.setter
    def orchestrator_llm(self, llm: BaseLLM) -> None:
        self._models.orchestrator_llm = llm

    @property
    def agent_llm(self) -> BaseLLM:
        """The decision model.  Owned by :class:`ModelRouter`."""
        return self._models.agent_llm

    @agent_llm.setter
    def agent_llm(self, llm: BaseLLM) -> None:
        self._models.agent_llm = llm

    def execute(
        self,
        user_message: str,
        mcp_servers: List[MCPServer],
        skills: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        document_context: Optional[str] = None,
        has_attachments: bool = False,
        project_dir: Optional[str] = None,
        model_service: Optional[Any] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        available_capabilities: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
        requested_skill: Optional[str] = None,
        current_date: Optional[str] = None,
    ) -> MCPExecutionResult:
        """Top-level entry point: plan → run → reason.

        ``progress_callback(step, status, message)`` — when supplied — is
        invoked as each step runs so the streaming layer can surface live
        activity to the client.  It fires from DAG worker threads.

        ``current_date`` — the client's wall-clock date ("Month DD, YYYY"),
        used to ground date-aware capability-step prompts.
        """
        self._models.begin_request(progress_callback)
        self._judge.begin_request()
        logger.info(
            "[models] response=%s  agent=%s  large=%s",
            self._models.model_label(self.orchestrator_llm),
            self._models.model_label(self.agent_llm),
            self._models.model_label(self._models.large_llm)
            if self._models.large_llm is not None
            else "(lazy)" if self._models.has_large_provider else "(none)",
        )
        try:
            ctx = self._build_context(
                user_message=user_message,
                mcp_servers=mcp_servers,
                skills=skills,
                cwd=cwd,
                document_context=document_context,
                has_attachments=has_attachments,
                project_dir=project_dir,
                model_service=model_service,
                conversation_history=conversation_history,
                available_capabilities=available_capabilities,
                requested_skill=requested_skill,
            )
            dag, handlers = self.plan(ctx)
            results = self.run(ctx, dag, handlers)
            return self.reason(ctx, results)

        except Exception as e:
            logger.error(f"MCP orchestration failed: {e}", exc_info=True)
            return MCPExecutionResult(handled=False)

    def _build_context(
        self,
        user_message: str,
        mcp_servers: List[MCPServer],
        skills: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        document_context: Optional[str] = None,
        has_attachments: bool = False,
        project_dir: Optional[str] = None,
        model_service: Optional[Any] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        available_capabilities: Optional[List[str]] = None,
        requested_skill: Optional[str] = None,
    ) -> _OrchestratorContext:
        """Build the shared orchestrator context from input parameters."""
        available_servers = self._steps.build_available_servers(mcp_servers)
        effective_cwd = os.path.expanduser(cwd or project_dir or os.getcwd())

        mcp_tool_contexts: List[MCPToolContext] = []
        for srv in available_servers:
            # Include the system tools server ("smartloop") here so its
            # built-in tools (file-read, file-write, bash, …) resolve through
            # the same ```mcp-block path as third-party servers like fetch.
            # Execution is server-agnostic: find_server/_call_tool dispatch
            # against the full mcp_servers list regardless of server type.
            mcp_tool_contexts.append(MCPToolContext(
                server_id=srv.get("id", ""),
                server_name=srv.get("name", "Unknown"),
                tools=srv.get("tools", []),
            ))

        enriched = self.explorer.enrich(
            available_mcp_servers=available_servers,
            conversation_history=conversation_history,
        )

        return _OrchestratorContext(
            user_message=user_message,
            mcp_servers=mcp_servers,
            available_servers=available_servers,
            effective_cwd=effective_cwd,
            skills=list(skills or []),
            requested_skill=requested_skill,
            document_context=document_context,
            project_dir=project_dir,
            model_service=model_service,
            conversation_history=conversation_history,
            has_attachments=has_attachments,
            mcp_tool_contexts=mcp_tool_contexts,
            enriched=enriched,
            available_capabilities=list(available_capabilities or []),
        )

    def plan(self, ctx: _OrchestratorContext) -> tuple:
        """Phase 1: Build the execution DAG from skills, tools, and sources."""
        dag = DAG()

        skill_match_node = SkillMatchNode()
        dag.add_node(skill_match_node)

        collect_node = CollectNode(depends_on=[skill_match_node])
        dag.add_node(collect_node)

        route_node = RouteNode(
            depends_on=[skill_match_node, collect_node],
        )
        dag.add_node(route_node)

        handlers = self._build_handlers(
            ctx,
            skill_match_node=skill_match_node,
            collect_node=collect_node,
            route_node=route_node,
        )

        return dag, handlers

    def run(self, ctx: _OrchestratorContext, dag: DAG, handlers: Dict[str, Callable]) -> Dict[Any, NodeResult]:
        """Phase 2: Walk the DAG and accumulate raw results from each node."""
        return self._dag_executor.execute(dag, handlers)

    def reason(self, ctx: _OrchestratorContext, results: Dict[Any, NodeResult]) -> MCPExecutionResult:
        """Phase 3: Analyze accumulated results and produce the final response."""
        fmt = results.get(REASON_NODE)
        if not (fmt and fmt.success and isinstance(fmt.data, dict)):
            return MCPExecutionResult(handled=False)

        accumulated = fmt.data.get("accumulated", [])
        search_ctx = fmt.data.get("search_ctx")
        is_conversation = fmt.data.get("is_conversation", False)
        synthesized = fmt.data.get("synthesized")

        if not search_ctx:
            return MCPExecutionResult(handled=False)

        tool_results = fmt.data.get("tool_results", [])
        tool_names_used = [r["tool_name"] for r in accumulated] if accumulated else []
        explore_citations = search_ctx.citations

        dag_citations = fmt.data.get("citations")
        if dag_citations:
            explore_citations = dag_citations

        cited_docs = getattr(search_ctx, "cited_documents", None)
        if cited_docs and explore_citations:
            allowed = set(cited_docs)
            explore_citations = [
                c for c in explore_citations
                if c.get("document_name") in allowed or not c.get("document_name")
            ]

        non_explore = [r["content"] for r in accumulated if r["tool_name"] != EXPLORE_TOOL]
        has_tool_output = any(c for c in non_explore)
        explore_parts = [r["content"] for r in accumulated if r["tool_name"] == EXPLORE_TOOL]
        has_explore_output = any(c for c in explore_parts)

        # Citations track which indexed documents contributed to the
        # response.  Include them when the explore step produced content
        # (documents were read and fed into the answer).  Suppress them
        # only when tool output is the sole data source and no documents
        # were part of the pipeline.
        keep_citations = has_explore_output

        if synthesized:
            return self._response.build(
                ctx=ctx,
                search_ctx=search_ctx,
                tool_names_used=tool_names_used,
                vector_context=None,
                tool_context=synthesized,
                explore_citations=explore_citations if keep_citations else None,
                is_conversation=is_conversation,
                is_synthesized=True,
            )

        tool_context = "\n\n---\n\n".join(non_explore) if non_explore else ""

        if tool_context:
            return self._response.build(
                ctx=ctx,
                search_ctx=search_ctx,
                tool_names_used=tool_names_used,
                vector_context=None,
                tool_context=tool_context,
                explore_citations=None,
                is_conversation=is_conversation,
            )

        vector_context = search_ctx.vector_context
        if explore_parts and not vector_context:
            vector_context = "\n\n---\n\n".join(explore_parts)

        return self._response.build(
            ctx=ctx,
            search_ctx=search_ctx,
            tool_names_used=tool_names_used,
            vector_context=vector_context,
            tool_context=tool_context,
            explore_citations=explore_citations,
            is_conversation=is_conversation,
        )

    def _build_handlers(
        self,
        ctx: _OrchestratorContext,
        skill_match_node: SkillMatchNode,
        collect_node: CollectNode,
        route_node: RouteNode,
    ) -> Dict[str, Callable]:
        """Build the handler map for the initial DAG nodes."""
        return {
            skill_match_node.node_id: lambda deps: self._skills.match(ctx, skill_match_node),
            collect_node.node_id: lambda deps: self._pipeline.handle_collect(
                ctx, deps, collect_node, skill_match_node,
            ),
            route_node.node_id: lambda deps: self._pipeline.handle_route(
                ctx, deps, route_node, skill_match_node, collect_node,
            ),
        }
