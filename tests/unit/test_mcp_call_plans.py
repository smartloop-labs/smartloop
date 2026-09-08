"""The second call: fetching what a listing only named.

`notion-search` answers with page titles and ids; the pitch document's text is
behind `notion-fetch`. Nothing in either schema says the two go together, so
the pairing is declared per server and applied to the call a question just
made — the surviving half of the crawl plans, now that nothing is crawled.

What matters here: a plan is found for the server that has one and nobody
else's, the detail tool and its id argument are resolved against what the
server actually declares, ids are read out of whatever envelope the payload
used, and every "no follow-up" case is silent rather than an error.
"""

from types import SimpleNamespace

import pytest

from smartloop.mcp.call_plans import (
    CallPlan,
    coerce_argument,
    filter_arguments,
    follow_up_calls,
    plan_for,
    plans,
    record_ids,
    resolve_argument,
    resolve_tool,
)

_SEARCH = {
    "name": "notion-search",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}
_FETCH = {
    "name": "notion-fetch",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "include_discussions": {"type": "boolean"},
        },
        "required": ["id"],
    },
}


def _server(name="Notion", catalog_slug="notion", tools=(_SEARCH, _FETCH), **fields):
    return SimpleNamespace(
        name=name, catalog_slug=catalog_slug, pkg=fields.get("pkg"),
        remote_config=fields.get("remote_config"), config=fields.get("config"),
        tools=list(tools),
    )


_HITS = (
    '{"results": [{"id": "abc-123", "title": "Pitch document"},'
    ' {"id": "def-456", "title": "Roadmap"}]}'
)


class TestTheBundledPlans:
    def test_the_shipped_plans_load(self):
        assert {"notion", "linear", "trello", "canva", "monday"} <= set(plans())

    def test_the_catalog_slug_finds_the_plan(self):
        assert plan_for(_server()).slug == "notion"

    def test_a_hand_added_connection_matches_on_its_name(self):
        found = plan_for(_server(name="my notion workspace", catalog_slug=None))
        assert found is not None and found.slug == "notion"

    def test_a_server_with_no_plan_has_none(self):
        assert plan_for(_server(name="weather", catalog_slug=None, tools=())) is None


class TestFollowUpCalls:
    def test_each_listed_record_gets_its_body_fetched(self):
        calls = follow_up_calls(_server(), "notion-search", _HITS)
        assert calls == [
            ("notion-fetch", {"include_discussions": True, "id": "abc-123"}),
            ("notion-fetch", {"include_discussions": True, "id": "def-456"}),
        ]

    def test_the_detail_call_does_not_follow_up_on_itself(self):
        assert follow_up_calls(_server(), "notion-fetch", _HITS) == []

    def test_a_server_without_the_detail_tool_gets_none(self):
        assert follow_up_calls(_server(tools=(_SEARCH,)), "notion-search", _HITS) == []

    def test_a_payload_with_no_records_is_not_an_error(self):
        assert follow_up_calls(_server(), "notion-search", "nothing to see") == []
        assert follow_up_calls(_server(), "notion-search", "") == []

    def test_a_connection_with_no_plan_is_not_an_error(self):
        server = _server(name="weather", catalog_slug=None)
        assert follow_up_calls(server, "notion-search", _HITS) == []

    def test_the_number_of_follow_ups_is_bounded(self):
        """Twenty round-trips to answer a question about one page is a worse
        answer, later."""
        many = '{"results": [%s]}' % ", ".join(
            f'{{"id": "id-{i}"}}' for i in range(20)
        )
        assert len(follow_up_calls(_server(), "notion-search", many)) == 3

    def test_a_fixed_argument_the_tool_rejects_is_dropped(self):
        plain = {
            "name": "notion-fetch",
            "input_schema": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        }
        calls = follow_up_calls(_server(tools=(_SEARCH, plain)), "notion-search", _HITS)
        assert calls[0] == ("notion-fetch", {"id": "abc-123"})


class TestReadingRecordsOutOfAPayload:
    @pytest.mark.parametrize("payload, expected", [
        ('{"results": [{"id": "a"}]}', ["a"]),
        ('[{"id": "a"}, {"id": "b"}]', ["a", "b"]),
        ('{"data": {"items": [{"id": "a"}]}}', ["a"]),
        ('{"edges": [{"node": {"id": "a"}}]}', ["a"]),
        ('{"id": "a", "title": "one record on its own"}', ["a"]),
        ('Here are your pages:\n{"results": [{"id": "a"}]}', ["a"]),
    ])
    def test_the_envelope_does_not_matter(self, payload, expected):
        assert record_ids(payload) == expected

    def test_a_url_is_an_id_of_last_resort(self):
        # Hosted Notion listings carry no id; the URL is the only handle.
        assert record_ids('{"results": [{"url": "https://notion.so/p/1"}]}') == [
            "https://notion.so/p/1"
        ]

    def test_the_id_key_a_server_actually_uses_is_found(self):
        assert record_ids('{"items": [{"design_id": "DAF1"}]}') == ["DAF1"]

    def test_duplicates_are_not_fetched_twice(self):
        assert record_ids('[{"id": "a"}, {"id": "a"}, {"id": "b"}]') == ["a", "b"]

    def test_unparseable_output_yields_nothing(self):
        assert record_ids("the server said no") == []
        assert record_ids("{not json") == []


class TestResolvingAgainstWhatTheServerDeclares:
    def test_a_plan_name_pairs_with_the_server_spelling(self):
        # A plan says `fetch`; the server calls it `notion-fetch`.
        assert resolve_tool(["notion-search", "notion-fetch"], ["fetch"]) == "notion-fetch"

    def test_candidate_order_decides(self):
        names = ["get_page", "notion-fetch"]
        assert resolve_tool(names, ["notion-fetch", "get_page"]) == "notion-fetch"

    def test_a_tool_the_server_lacks_costs_nothing(self):
        assert resolve_tool(["notion-search"], ["notion-fetch"]) is None

    def test_the_id_argument_comes_from_the_schema(self):
        schema = {"properties": {"pageId": {"type": "string"}}}
        assert resolve_argument(schema, ["page_id", "pageId"]) == "pageId"

    def test_a_lone_required_argument_takes_the_id_whatever_it_is_called(self):
        schema = {"properties": {"ref": {"type": "string"}}, "required": ["ref"]}
        assert resolve_argument(schema, ["id", "page_id"]) == "ref"

    def test_arguments_the_tool_does_not_declare_are_dropped(self):
        schema = {"properties": {"id": {"type": "string"}}}
        assert filter_arguments(schema, {"id": "a", "nonsense": 1}) == {"id": "a"}

    def test_an_empty_schema_is_no_evidence_so_nothing_is_dropped(self):
        assert filter_arguments({}, {"id": "a"}) == {"id": "a"}

    def test_a_numeric_id_argument_is_cast(self):
        # monday returns board ids as strings and declares them as numbers.
        schema = {"properties": {"boardId": {"type": "number"}}}
        assert coerce_argument(schema, "boardId", "12345") == 12345
        assert coerce_argument(schema, "boardId", "not-a-number") == "not-a-number"

    def test_a_string_id_is_left_alone(self):
        schema = {"properties": {"id": {"type": "string"}}}
        assert coerce_argument(schema, "id", "abc-123") == "abc-123"


class TestPlanValidation:
    def test_a_misspelled_field_is_rejected(self):
        with pytest.raises(Exception, match="detail_tool"):
            CallPlan(slug="x", detail_tool=["get"])
