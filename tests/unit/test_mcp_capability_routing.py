"""An MCP tool call follows the step's capability, like every other node.

``pipeline_builder`` tags a non-skill TOOL step ``SMALL`` and *still* sets its
capability, but :meth:`ModelRouter.llm_for_tier` only consulted the capability
provider on the ``LARGE`` branch — so the tag was read by nothing and both
halves of an MCP call (picking the tool, extracting its arguments) ran on the
bootstrap router.  It only showed with two or more models selected: a single
selection is pinned into ``agent_llm`` upstream, so the default *was* the
user's model.

SMALL routes best-effort — ``decision_llm`` takes the capability model only
when it is resident or fits — so honouring the tag never forces a load onto a
memory-tight device.
"""

from unittest.mock import MagicMock

from smartloop.agents.mcp.mcp_agent import MCPAgent
from smartloop.agents.orchestration.model_router import ModelRouter
from smartloop.agents.planner import ModelTier
from smartloop.types import ModelCapability


def _llm(name):
    llm = MagicMock()
    llm.model.metadata = {"general.name": name, "general.size_label": ""}
    return llm


def _router(*, capability_llm=None):
    agent_llm = _llm("Slp Orchestra Mini")
    router = ModelRouter(
        orchestrator_llm=_llm("Slp Orchestra Mini"),
        agent_llm=agent_llm,
        capability_llm_provider=lambda cap: capability_llm,
        capability_available=lambda cap: capability_llm is not None,
    )
    router.begin_request(MagicMock())
    return router, agent_llm


class TestSmallTierHonoursCapability:
    def test_a_tagged_small_step_gets_the_capability_model(self):
        cap_llm = _llm("Gemma 3n E2B")
        router, agent_llm = _router(capability_llm=cap_llm)

        resolved = router.llm_for_tier(ModelTier.SMALL, ModelCapability.CODING)

        assert resolved is cap_llm
        assert resolved is not agent_llm

    def test_an_untagged_small_step_still_gets_the_agent_model(self):
        router, agent_llm = _router(capability_llm=_llm("Gemma 3n E2B"))

        assert router.llm_for_tier(ModelTier.SMALL) is agent_llm

    def test_it_falls_back_when_the_capability_model_will_not_load(self):
        """decision_llm never forces a load — a declined provider lands here."""
        router, agent_llm = _router(capability_llm=None)

        resolved = router.llm_for_tier(ModelTier.SMALL, ModelCapability.CODING)

        assert resolved is agent_llm

    def test_a_provider_failure_does_not_break_the_step(self):
        agent_llm = _llm("Slp Orchestra Mini")
        router = ModelRouter(
            orchestrator_llm=agent_llm,
            agent_llm=agent_llm,
            capability_llm_provider=MagicMock(side_effect=RuntimeError("oom")),
        )
        router.begin_request(MagicMock())

        assert router.llm_for_tier(ModelTier.SMALL, ModelCapability.DP) is agent_llm

    def test_the_large_branch_is_unchanged(self):
        cap_llm = _llm("Gemma 3n E2B")
        router, _agent_llm = _router(capability_llm=cap_llm)

        assert router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS) is cap_llm


class TestToolResolutionTakesTheOverride:
    """The pick is the harder judgement of the two — upgrading only the
    argument extraction just got the arguments right for the wrong tool."""

    @staticmethod
    def _agent_with_two_tools(default_llm):
        agent = MCPAgent(default_llm)
        server = MagicMock()
        server.id = "srv-1"
        server.name = "linear"
        server.enabled = True
        server.system = False
        server.tools = [
            {"name": "list_issues", "description": "List issues",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "create_issue", "description": "Create an issue",
             "input_schema": {
                 "type": "object",
                 "properties": {"title": {"type": "string",
                                          "description": "The issue title"}},
                 "required": ["title"],
             }},
        ]
        return agent, [server]

    def test_the_override_model_resolves_the_call(self):
        default_llm = _llm("Slp Orchestra Mini")
        default_llm.invoke = MagicMock(return_value='{"tool": 2}')
        capability_llm = _llm("Gemma 3n E2B")
        capability_llm.invoke = MagicMock(return_value='{"tool": 1}')
        agent, servers = self._agent_with_two_tools(default_llm)

        resolved = agent.resolve(
            directive="list my open issues",
            mcp_servers=servers,
            llm=capability_llm,
        )

        assert (resolved.server_id, resolved.tool_name) == ("srv-1", "list_issues")
        capability_llm.invoke.assert_called_once()
        default_llm.invoke.assert_not_called()

    def test_without_an_override_the_agent_model_still_resolves(self):
        default_llm = _llm("Slp Orchestra Mini")
        default_llm.invoke = MagicMock(
            return_value='{"tool": 2, "arguments": {"title": "Login is broken"}}'
        )
        agent, servers = self._agent_with_two_tools(default_llm)

        resolved = agent.resolve(
            directive="file an issue: Login is broken", mcp_servers=servers,
        )

        assert (resolved.server_id, resolved.tool_name) == ("srv-1", "create_issue")
        # The pick and its required argument come out of the same call.
        assert resolved.arguments == {"title": "Login is broken"}
        default_llm.invoke.assert_called_once()


class TestTheReasoningModelIsAnnouncedAsWork:
    """What the user is told about the model that resolves a tool call.

    ``thinking`` is an internal routing tag: the API leaves it out of the
    capabilities it reports, and it names a property of the model rather than
    the work in progress.  Surfaced verbatim it read "Selected qwen3.5-2b for
    thinking" — and on the single-model workspace where that model also writes
    the answer, it landed directly beside "Selected qwen3.5-2b for the
    response": one model, twice, under two labels.
    """

    def _router(self, capability_llm):
        agent_llm = _llm("Qwen3.5 2B")
        agent_llm.supports_thinking = False
        progress = MagicMock()
        router = ModelRouter(
            orchestrator_llm=agent_llm,
            agent_llm=agent_llm,
            capability_llm_provider=lambda cap: capability_llm,
            capability_available=lambda cap: capability_llm is not None,
        )
        router.begin_request(progress)
        return router, agent_llm, progress

    def _messages(self, progress):
        return [call.args[2] for call in progress.call_args_list]

    def test_a_switch_is_announced_as_tool_resolution(self):
        reasoner = _llm("Qwen3 4B")
        reasoner.supports_thinking = True
        router, agent_llm, progress = self._router(reasoner)

        assert router.reasoning_llm(agent_llm) is reasoner

        messages = self._messages(progress)
        assert messages == ["Selected Qwen3 4B for tool resolution"]
        assert not any("thinking" in m for m in messages)

    def test_landing_on_the_model_already_in_use_is_not_announced(self):
        """Nothing was switched, so there is nothing to report."""
        agent_llm = _llm("Qwen3.5 2B")
        agent_llm.supports_thinking = False
        progress = MagicMock()
        router = ModelRouter(
            orchestrator_llm=agent_llm,
            agent_llm=agent_llm,
            # The workspace's reasoning model *is* the model this request runs
            # on — the single-model case.
            capability_llm_provider=lambda cap: agent_llm,
            capability_available=lambda cap: True,
        )
        router.begin_request(progress)

        router.reasoning_llm(agent_llm)

        assert self._messages(progress) == []

    def test_a_fallback_that_already_reasons_is_kept_and_silent(self):
        reasoner = _llm("Qwen3 4B")
        reasoner.supports_thinking = True
        router, _agent_llm, progress = self._router(_llm("Gemma 3n E2B"))

        assert router.reasoning_llm(reasoner) is reasoner
        assert self._messages(progress) == []

    def test_no_reasoning_model_in_the_workspace_says_nothing(self):
        router, agent_llm, progress = self._router(None)

        assert router.reasoning_llm(agent_llm) is agent_llm
        assert self._messages(progress) == []

    def _multi_model_router(self, capability_llm):
        """Two or more models selected: the request itself runs on the small
        router model, so a capability model is always a real switch."""
        router_llm = _llm("Slp Orchestra Mini")
        router_llm.supports_thinking = False
        progress = MagicMock()
        router = ModelRouter(
            orchestrator_llm=router_llm,
            agent_llm=router_llm,
            capability_llm_provider=lambda cap: capability_llm(cap),
            capability_available=lambda cap: capability_llm(cap) is not None,
        )
        router.begin_request(progress)
        return router, router_llm, progress

    def test_a_model_already_announced_is_not_announced_again(self):
        """One model serving two capabilities is one selection, not two.

        Synthesis names it for ``skills``; the tool call lands on the same
        model a moment later.  Both are switches away from the router model,
        so neither is suppressed by the "already in use" check — and the user
        read the same model being picked twice:

            Selected qwen3.5-2b for skills
            Selected qwen3.5-2b for tool resolution
        """
        worker = _llm("Qwen3.5 2B")
        worker.supports_thinking = True
        router, router_llm, progress = self._multi_model_router(lambda cap: worker)

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)
        router.reasoning_llm(router_llm)

        selected = [m for m in self._messages(progress) if m.startswith("Selected")]
        assert selected == ["Selected Qwen3.5 2B for skills"]

    def test_a_different_model_for_the_tool_call_is_still_announced(self):
        """Suppression is about repetition, not about the second event: work
        moving to another model is exactly what the line is for."""
        worker = _llm("Qwen3.5 2B")
        worker.supports_thinking = False
        reasoner = _llm("Qwen3 4B")
        reasoner.supports_thinking = True
        router, router_llm, progress = self._multi_model_router(
            lambda cap: reasoner if cap == ModelCapability.THINKING.value else worker,
        )

        router.llm_for_tier(ModelTier.LARGE, ModelCapability.SKILLS)
        router.reasoning_llm(router_llm)

        selected = [m for m in self._messages(progress) if m.startswith("Selected")]
        assert selected == [
            "Selected Qwen3.5 2B for skills",
            "Selected Qwen3 4B for tool resolution",
        ]
