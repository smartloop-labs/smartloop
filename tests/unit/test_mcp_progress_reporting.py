"""Collect reports the MCP calls it makes, not just that it is busy.

A question answered from a connected server spends most of its collect inside
MCP: a reasoned tool pick over the server's catalogue, the call, then one
follow-up fetch per record.  All of it used to sit behind a single
"Processing..." — web search announced itself, MCP did not — so a request that
was reading three Notion pages was indistinguishable from a hang.
"""

from unittest.mock import MagicMock

from smartloop.agents.mcp.explorer import Explorer


def _explorer():
    exp = Explorer.__new__(Explorer)
    exp.mcp_agent = MagicMock()
    exp.tool_caller = MagicMock()
    exp.arg_extractor = MagicMock()
    return exp


def _server(name="Notion"):
    s = MagicMock()
    s.name = name
    return s


class TestTheResolvedCallIsReported:
    def test_the_long_resolution_leg_announces_itself(self):
        exp = _explorer()
        exp.mcp_agent.resolve.return_value = None
        seen = []
        exp._run_mcp_server("q", _server(), progress=lambda *a: seen.append(a))
        assert seen, "resolution reported nothing"
        step, status, message = seen[0]
        assert (step, status) == ("tool", "processing")
        assert "Notion" in message

    def test_the_tool_and_its_arguments_are_named(self):
        # "Calling Notion" cannot tell reading a page from searching for one,
        # and which ran is the thing a user watching this wants to know.
        exp = _explorer()
        picked = MagicMock()
        picked.tool_name = "notion-search"
        picked.arguments = {"query": "founders agreement"}
        exp.mcp_agent.resolve.return_value = picked
        exp._call_mcp_tool = MagicMock(return_value="some records")
        exp._fetch_record_details = MagicMock(return_value="")

        seen = []
        exp._run_mcp_server("q", _server(), progress=lambda *a: seen.append(a))

        messages = [m for _, _, m in seen]
        assert any("notion-search" in m for m in messages), messages
        assert any("founders agreement" in m for m in messages), messages

    def test_a_write_tool_left_to_an_execute_step_reports_no_call(self):
        exp = _explorer()
        picked = MagicMock()
        picked.tool_name = "notion-create-pages"
        picked.arguments = {}
        exp.mcp_agent.resolve.return_value = picked

        seen = []
        exp._run_mcp_server("q", _server(), progress=lambda *a: seen.append(a))

        assert not any("Calling" in m for _, _, m in seen)


class TestTheFollowUpFetchesAreReported:
    def test_each_record_read_is_reported(self, monkeypatch):
        exp = _explorer()
        result = MagicMock()
        result.success = True
        result.content = [{"type": "text", "text": "the page body"}]
        exp.tool_caller.call_tool.return_value = result

        monkeypatch.setattr(
            "smartloop.mcp.call_plans.follow_up_calls",
            lambda server, tool, payload: [
                ("notion-fetch", {"id": "aaa"}),
                ("notion-fetch", {"id": "bbb"}),
            ],
        )

        seen = []
        exp._fetch_record_details(
            _server(), "notion-search", "payload",
            progress=lambda *a: seen.append(a),
        )

        messages = [m for _, _, m in seen]
        assert len(messages) == 2, messages
        assert "1 of 2" in messages[0] and "2 of 2" in messages[1]
        assert all("notion-fetch" in m for m in messages)


class TestReportingNeverBreaksCollect:
    def test_a_raising_callback_does_not_fail_the_call(self):
        # The callback hops into the request's event loop, so it can raise long
        # after the client has gone.  Collect's job is the context.
        exp = _explorer()
        picked = MagicMock()
        picked.tool_name = "notion-search"
        picked.arguments = {"query": "x"}
        exp.mcp_agent.resolve.return_value = picked
        exp._call_mcp_tool = MagicMock(return_value="records")
        exp._fetch_record_details = MagicMock(return_value="")

        def boom(*_a):
            raise RuntimeError("client is gone")

        out = exp._run_mcp_server("q", _server(), progress=boom)
        assert "notion-search" in out

    def test_no_callback_is_fine(self):
        exp = _explorer()
        exp.mcp_agent.resolve.return_value = None
        assert exp._run_mcp_server("q", _server()) == ""
