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


class TestToolPickTakesTheOverride:
    """The pick is the harder judgement of the two — upgrading only the
    argument extraction just got the arguments right for the wrong tool."""

    @staticmethod
    def _agent_with_two_candidates(default_llm):
        agent = MCPAgent(default_llm)
        # Two candidates, so the pick actually goes to the LLM (a single
        # candidate needs no disambiguation and short-circuits).
        candidates = []
        for i, (server_id, tool) in enumerate(
            [("srv-1", "list_issues"), ("srv-1", "create_issue")]
        ):
            c = MagicMock()
            c.metadata = {
                "server_id": server_id,
                "tool_name": tool,
                "qualified_name": f"linear.{tool}",
            }
            c.text = tool
            candidates.append(c)
        server = MagicMock()
        server.id = "srv-1"
        server.enabled = True
        server.tools = []
        return agent, candidates, [server]

    def _select(self, monkeypatch, agent, candidates, servers, **kwargs):
        indexer = MagicMock()
        indexer.search.return_value = candidates
        monkeypatch.setattr(
            "smartloop.db.mcp_tool_indexer.MCPToolIndexer",
            MagicMock(return_value=indexer),
        )
        return agent.select_tool(
            directive="list my open issues",
            project_dir="/tmp/project",
            mcp_servers=servers,
            **kwargs,
        )

    def test_the_override_model_makes_the_pick(self, monkeypatch):
        default_llm = MagicMock(invoke=MagicMock(return_value="2"))
        capability_llm = MagicMock(invoke=MagicMock(return_value="1"))
        agent, candidates, servers = self._agent_with_two_candidates(default_llm)

        picked = self._select(
            monkeypatch, agent, candidates, servers, llm=capability_llm,
        )

        assert picked == ("srv-1", "list_issues")
        capability_llm.invoke.assert_called_once()
        default_llm.invoke.assert_not_called()

    def test_without_an_override_the_agent_model_still_picks(self, monkeypatch):
        default_llm = MagicMock(invoke=MagicMock(return_value="2"))
        agent, candidates, servers = self._agent_with_two_candidates(default_llm)

        picked = self._select(monkeypatch, agent, candidates, servers)

        assert picked == ("srv-1", "create_issue")
        default_llm.invoke.assert_called_once()
