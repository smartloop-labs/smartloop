"""Resolving an MCP step against what discovery already found.

Selection used to be a vector search over embedded tool descriptions, which
answered "which tool reads like this sentence" and then handed the pick to a
second call that had to work out the arguments on its own.  Resolution is now
one call over the tools discovery recorded per server: it names the tool *and*
fills its required arguments, and a reasoning model is given an output budget
big enough to think first.

These pin the parts that are easy to regress: the catalog comes from the
servers in memory (no index, no project dir), the reply is parsed in the shapes
models actually emit, fabricated values are dropped, and a thinking model asks
for a budget above the floor below which the wrapper suppresses the block.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.agents.mcp.mcp_agent import (
    RESOLVE_MAX_TOKENS, MCPAgent, clear_resolution_cache, format_call,
    missing_required,
)
from smartloop.agents.planner import SYSTEM_TOOLS_SERVER_ID
from smartloop.llms.llama_cpp import THINKING_MIN_BUDGET


@pytest.fixture(autouse=True)
def _fresh_cache():
    """One question resolves once per request — including across these tests."""
    clear_resolution_cache()
    yield
    clear_resolution_cache()


def _llm(reply: str, *, can_think: bool = True):
    """An LLM that answers *reply*.

    ``can_think`` is what both real handles report: ``LlamaCpp`` from the flag
    bound onto its ``Llama`` at load, ``RemoteLLM`` by asking the host.  Its
    *value* — the user's setting for visible answers — is deliberately not
    represented here, because resolution does not consult it.
    """
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=reply)
    llm.supports_thinking = can_think
    llm.model_label = "test-model"
    # get_max_tokens/get_num_tokens stay MagicMocks: the budget helper falls
    # back to what it asked for when the arithmetic isn't numeric, which is
    # what a caller with no real tokenizer should see.
    return llm


def _server(server_id="srv-1", name="linear", tools=None, **fields):
    server = MagicMock()
    server.id = server_id
    server.name = name
    server.enabled = fields.get("enabled", True)
    server.system = fields.get("system", False)
    server.tools = tools if tools is not None else [_tool()]
    return server


def _tool(name="create_issue", description="Create an issue", required=("title",)):
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The issue title"},
                "teamId": {"type": "string", "description": "Team the issue belongs to"},
                "after": {"type": "string", "description": "Pagination cursor"},
            },
            "required": list(required),
        },
    }


class TestTheCatalogIsDiscovery:
    def test_the_prompt_lists_every_discovered_tool_with_its_required_args(self):
        llm = _llm('{"tool": 1, "arguments": {"title": "Login is broken"}}')
        agent = MCPAgent(llm)

        resolved = agent.resolve(
            directive="file an issue: Login is broken",
            mcp_servers=[_server(tools=[_tool(), _tool(name="list_issues", required=())])],
        )

        prompt = llm.invoke.call_args.args[0]
        assert "linear.create_issue" in prompt
        assert "linear.list_issues" in prompt
        assert "title (string): The issue title" in prompt
        # A tool with no required arguments says so rather than listing the
        # optional ones — an optional argument the model can see is one it can
        # fabricate.
        assert "(no required arguments)" in prompt
        assert "after (string)" not in prompt
        assert resolved.tool_name == "create_issue"

    def test_it_resolves_with_no_project_dir_and_no_index(self):
        """No vector store, no embeddings, nothing on disk — just the servers."""
        agent = MCPAgent(_llm('{"tool": 1, "arguments": {"title": "x"}}'))

        resolved = agent.resolve(directive="file an issue", mcp_servers=[_server()])

        assert resolved.server_id == "srv-1"

    def test_system_and_disabled_servers_are_left_out(self):
        llm = _llm('{"tool": 0}')
        agent = MCPAgent(llm)

        agent.resolve(
            directive="anything",
            mcp_servers=[
                _server(server_id=SYSTEM_TOOLS_SERVER_ID, name="smartloop",
                        tools=[_tool(name="bash")]),
                _server(server_id="srv-2", name="off", enabled=False,
                        tools=[_tool(name="get_page")]),
                _server(),
            ],
        )

        prompt = llm.invoke.call_args.args[0]
        assert "bash" not in prompt
        assert "off.get_page" not in prompt
        assert "linear.create_issue" in prompt

    def test_nothing_discovered_means_no_call_and_no_inference(self):
        llm = _llm('{"tool": 1}')
        agent = MCPAgent(llm)

        assert agent.resolve(directive="anything", mcp_servers=[]) is None
        assert agent.resolve(directive="anything", mcp_servers=[_server(tools=[])]) is None
        llm.invoke.assert_not_called()

    def test_a_lone_argument_free_tool_needs_no_model_at_all(self):
        llm = _llm('{"tool": 1}')
        agent = MCPAgent(llm)

        resolved = agent.resolve(
            directive="what is on my board",
            mcp_servers=[_server(tools=[_tool(name="list_issues", required=())])],
        )

        assert resolved.tool_name == "list_issues"
        assert resolved.arguments == {}
        llm.invoke.assert_not_called()


class TestParsingWhatModelsActuallyEmit:
    def _resolve(self, reply):
        agent = MCPAgent(_llm(reply))
        return agent.resolve(
            directive="file an issue: Login is broken",
            mcp_servers=[_server(tools=[_tool(name="list_issues", required=()), _tool()])],
        )

    def test_the_json_object_it_was_asked_for(self):
        resolved = self._resolve('{"tool": 2, "arguments": {"title": "Login is broken"}}')
        assert resolved.tool_name == "create_issue"
        assert resolved.arguments == {"title": "Login is broken"}

    def test_a_fenced_object(self):
        resolved = self._resolve('```json\n{"tool": 2, "arguments": {"title": "t"}}\n```')
        assert resolved.tool_name == "create_issue"

    def test_the_tool_named_instead_of_numbered(self):
        resolved = self._resolve('{"tool": "linear.create_issue", "arguments": {"title": "t"}}')
        assert resolved.tool_name == "create_issue"

    def test_a_bare_number_from_a_model_that_ignored_the_shape(self):
        resolved = self._resolve("2")
        assert resolved.tool_name == "create_issue"
        assert resolved.arguments == {}

    def test_zero_means_none_of_them_fit(self):
        assert self._resolve('{"tool": 0}') is None

    def test_an_out_of_range_number_is_not_a_pick(self):
        assert self._resolve('{"tool": 9}') is None

    def test_unparseable_output_is_not_a_pick(self):
        assert self._resolve("I could not find a suitable tool.") is None

    def test_a_failed_call_is_a_skip_not_a_raise(self):
        llm = _llm("")
        llm.invoke.side_effect = RuntimeError("oom")
        agent = MCPAgent(llm)

        assert agent.resolve(directive="anything", mcp_servers=[_server()]) is None


class TestArgumentsAreCleaned:
    def _arguments(self, raw_arguments):
        agent = MCPAgent(_llm('{"tool": 1, "arguments": %s}' % raw_arguments))
        resolved = agent.resolve(
            directive="file an issue: Login is broken", mcp_servers=[_server()],
        )
        return resolved.arguments

    def test_undeclared_and_unlisted_arguments_are_dropped(self):
        # `teamId` is declared but not required, `nonsense` is not declared at
        # all: neither was shown to the model, so neither is passed on.
        assert self._arguments(
            '{"title": "Login is broken", "teamId": "T-1", "nonsense": 1}'
        ) == {"title": "Login is broken"}

    def test_placeholders_and_empties_are_dropped(self):
        assert self._arguments('{"title": "MISSING"}') == {}
        assert self._arguments('{"title": ""}') == {}
        assert self._arguments('{"title": null}') == {}

    def test_an_echoed_field_description_is_not_a_value(self):
        assert self._arguments('{"title": "The issue title"}') == {}

    def test_a_cursor_is_never_accepted_from_the_model(self):
        agent = MCPAgent(_llm('{"tool": 1, "arguments": {"after": "cursor-123"}}'))
        resolved = agent.resolve(
            directive="next page",
            mcp_servers=[_server(tools=[_tool(required=("after",))])],
        )
        assert resolved.arguments == {}


class TestReasoningGetsRoomToThink:
    def test_the_call_asks_for_reasoning_and_room_to_do_it(self):
        """``thinking=True`` travels with the request, and the budget clears
        THINKING_MIN_BUDGET — below that the wrapper suppresses the block, so a
        resolve call would spend its budget on a thought it never finishes."""
        llm = _llm('{"tool": 1, "arguments": {"title": "Login is broken"}}')
        agent = MCPAgent(llm)

        resolved = agent.resolve(directive="file an issue", mcp_servers=[_server()])

        assert llm.invoke.call_args.kwargs["thinking"] is True
        assert llm.invoke.call_args.kwargs["max_tokens"] == RESOLVE_MAX_TOKENS
        assert RESOLVE_MAX_TOKENS > THINKING_MIN_BUDGET
        assert resolved.reasoned is True

    def test_reasoning_is_asked_for_whatever_the_users_setting_says(self):
        """The thinking toggle governs the visible answer; this call's output is
        a JSON tool call whose monologue is stripped either way.  So the flag is
        passed per call rather than read off the model."""
        llm = _llm('{"tool": 1, "arguments": {"title": "Login is broken"}}')
        agent = MCPAgent(llm)

        agent.resolve(directive="file an issue", mcp_servers=[_server()])

        # Nothing about the model's own setting is consulted — no attribute
        # read, no flag flipped: the request carries it.
        assert llm.invoke.call_args.kwargs["thinking"] is True

    def test_a_model_that_cannot_reason_still_resolves(self):
        llm = _llm('{"tool": 1, "arguments": {"title": "Login is broken"}}',
                   can_think=False)
        agent = MCPAgent(llm)

        resolved = agent.resolve(directive="file an issue", mcp_servers=[_server()])

        assert resolved.tool_name == "create_issue"
        assert resolved.reasoned is False

    def test_an_llm_without_the_per_call_flag_is_still_asked(self):
        """A handle whose invoke predates the flag must not lose the pick."""
        llm = _llm("")
        calls = []

        def answer(prompt, max_tokens=None, **kwargs):
            if kwargs:
                raise TypeError("invoke() got an unexpected keyword argument")
            calls.append(max_tokens)
            return '{"tool": 1, "arguments": {"title": "Login is broken"}}'

        llm.invoke.side_effect = answer
        agent = MCPAgent(llm)

        resolved = agent.resolve(directive="file an issue", mcp_servers=[_server()])

        assert resolved.tool_name == "create_issue"
        assert calls == [RESOLVE_MAX_TOKENS]

    def test_the_budget_never_exceeds_what_is_left_of_the_context(self):
        llm = _llm('{"tool": 1, "arguments": {"title": "t"}}', )
        llm.get_max_tokens = MagicMock(return_value=2048)
        llm.get_num_tokens = MagicMock(return_value=1800)
        agent = MCPAgent(llm)

        agent.resolve(directive="file an issue", mcp_servers=[_server()])

        assert llm.invoke.call_args.kwargs["max_tokens"] == 2048 - 1800 - 64


class TestTheCallIsShownToThePerson:
    """A tool step's decision is which call to make; until this was surfaced the
    client only saw which model had been selected."""

    def test_it_renders_the_call_as_a_signature(self):
        assert format_call("linear", "create_issue", {"title": "Login is broken"}) == (
            "linear.create_issue(title='Login is broken')"
        )

    def test_no_arguments_still_reads_as_a_call(self):
        assert format_call("linear", "list_issues", {}) == "linear.list_issues()"

    def test_a_long_value_is_truncated_not_pasted(self):
        shown = format_call("notion", "create-pages", {"content": "x" * 500})
        assert len(shown) < 120 and shown.endswith("…')")

    def test_extra_arguments_are_counted_rather_than_listed(self):
        shown = format_call("a", "b", {str(i): i for i in range(7)})
        assert shown.endswith("+3 more)")


class TestMissingRequired:
    def test_it_names_what_the_tool_still_needs(self):
        schema = {"required": ["title", "teamId"]}
        assert missing_required(schema, {"title": "t"}) == ["teamId"]
        assert missing_required(schema, {"title": "t", "teamId": ""}) == ["teamId"]
        assert missing_required(schema, {"title": "t", "teamId": "T-1"}) == []

    def test_a_tool_with_no_required_arguments_is_never_missing_any(self):
        assert missing_required({}, {}) == []


class TestOneQuestionResolvesOnce:
    """The collect path and the planned step both resolve the same question
    over the same catalog; the second would spend seconds of reasoning on an
    answer already in hand."""

    def test_the_second_caller_reuses_the_first_resolution(self):
        llm = _llm('{"tool": 1, "arguments": {"title": "Login is broken"}}')
        servers = [_server()]

        first = MCPAgent(llm).resolve("file an issue", servers)
        # A different MCPAgent instance — the two callers hold their own.
        second = MCPAgent(llm).resolve("file an issue", servers)

        assert second == first
        assert llm.invoke.call_count == 1

    def test_a_different_question_resolves_again(self):
        llm = _llm('{"tool": 1, "arguments": {"title": "Login is broken"}}')
        agent = MCPAgent(llm)

        agent.resolve("file an issue", [_server()])
        agent.resolve("file a different issue", [_server()])

        assert llm.invoke.call_count == 2

    def test_a_changed_catalog_resolves_again(self):
        llm = _llm('{"tool": 1, "arguments": {"title": "Login is broken"}}')
        agent = MCPAgent(llm)

        agent.resolve("file an issue", [_server()])
        agent.resolve("file an issue", [_server(tools=[_tool(), _tool(name="x")])])

        assert llm.invoke.call_count == 2

    def test_no_tool_is_remembered_too(self):
        """A question no tool serves must not be re-reasoned by the next
        caller either."""
        llm = _llm('{"tool": 0}')
        servers = [_server()]

        assert MCPAgent(llm).resolve("what time is it", servers) is None
        assert MCPAgent(llm).resolve("what time is it", servers) is None
        assert llm.invoke.call_count == 1


class TestADeclinedReplyIsSalvaged:
    """A 7.5B answered `{"tool": null, "arguments": {"context_note": "the tools
    are Notion APIs…"}}` to a question its Notion server plainly served.  Null
    is not a refusal the prompt offers — 0 is — so a name in the reply is taken
    rather than throwing the whole reasoning pass away."""

    def _entries(self):
        return [_server(tools=[
            _tool(name="notion-search", required=("title",)),
            _tool(name="notion-create-pages"),
        ])]

    def test_a_null_pick_that_names_a_tool_uses_it(self):
        agent = MCPAgent(_llm(
            '{"tool": null, "arguments": {"title": "the pitch"},'
            ' "note": "notion-search would find it"}'
        ))

        resolved = agent.resolve("outline the pitch from notion", self._entries())

        assert resolved.tool_name == "notion-search"
        assert resolved.arguments == {"title": "the pitch"}

    def test_an_explicit_zero_is_still_a_refusal(self):
        # The model is allowed to say no; only an unusable answer is salvaged.
        agent = MCPAgent(_llm('{"tool": 0, "arguments": {}}'))
        assert agent.resolve("what is 2 + 2", self._entries()) is None

    def test_prose_naming_no_tool_resolves_nothing(self):
        agent = MCPAgent(_llm("None of the available tools can do this."))
        assert agent.resolve("what is 2 + 2", self._entries()) is None


class TestTheCatalogIsSizedToTheModel:
    def test_a_roomy_context_keeps_the_descriptions(self):
        """Descriptions are what the pick is made on; a bare name list is the
        fallback for a small window, not the default."""
        llm = _llm('{"tool": 1, "arguments": {"title": "Login is broken"}}')
        llm.get_max_tokens = MagicMock(return_value=13056)
        tools = [_tool(name=f"notion-tool-{i}", description="Does a thing. " * 12)
                 for i in range(42)]
        MCPAgent(llm).resolve("file an issue", [_server(tools=tools)])

        prompt = llm.invoke.call_args.args[0]
        assert "Does a thing." in prompt
        assert "notion-tool-41" in prompt

    def test_a_small_context_falls_back_to_call_shapes(self):
        llm = _llm('{"tool": 1, "arguments": {"title": "Login is broken"}}')
        llm.get_max_tokens = MagicMock(return_value=2048)
        tools = [_tool(name=f"notion-tool-{i}", description="Does a thing. " * 12)
                 for i in range(42)]
        MCPAgent(llm).resolve("file an issue", [_server(tools=tools)])

        prompt = llm.invoke.call_args.args[0]
        assert "Does a thing." not in prompt
        # Every tool is still listed — names are dropped last, descriptions first.
        assert "notion-tool-41" in prompt
