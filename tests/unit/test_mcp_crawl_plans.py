"""Crawl plan resolution and tool matching.

A plan only fires if it can find the right tool on the server, and only the
document-type servers should be crawled by default — otherwise a shell or
weather connection gets walked every refresh cycle for nothing.
"""

from types import SimpleNamespace

from smartloop.mcp.crawl_plans import (
    LINEAR_PLAN,
    NOTION_PLAN,
    TRELLO_PLAN,
    filter_arguments,
    generic_plan,
    resolve_argument,
    resolve_plan,
    resolve_tool,
    should_crawl,
)


def _tool(name, schema=None):
    return SimpleNamespace(name=name, description=name, input_schema=schema or {})


def _server(name="demo", tools=(), **fields):
    fields.setdefault("catalog_slug", None)
    fields.setdefault("pkg", None)
    fields.setdefault("enabled", True)
    fields.setdefault("system", False)
    fields.setdefault("crawl", None)
    fields.setdefault("remote_config", None)
    return SimpleNamespace(id="s1", name=name, tools=list(tools), **fields)


class TestPlanResolution:
    def test_catalog_slug_selects_the_curated_plan(self):
        server = _server(name="My Workspace", catalog_slug="notion")
        assert resolve_plan(server) is NOTION_PLAN

    def test_hand_added_server_matches_on_name(self):
        assert resolve_plan(_server(name="linear-mcp")) is LINEAR_PLAN

    def test_match_can_come_from_the_remote_url(self):
        server = _server(
            name="tasks",
            remote_config=SimpleNamespace(server_url="https://mcp.trello.com/mcp"),
        )
        assert resolve_plan(server) is TRELLO_PLAN

    def test_unknown_server_falls_back_to_a_derived_plan(self):
        plan = resolve_plan(_server(name="weather", tools=[_tool("get_forecast")]))
        assert plan.derived


class TestShouldCrawl:
    def test_document_type_server_crawls_by_default(self):
        assert should_crawl(_server(name="notion", catalog_slug="notion")) is True

    def test_unknown_server_is_not_crawled_unless_asked(self):
        server = _server(name="weather", tools=[_tool("list_stations")])
        assert should_crawl(server) is False
        server.crawl = True
        assert should_crawl(server) is True

    def test_explicit_opt_out_wins_over_a_curated_plan(self):
        server = _server(name="notion", catalog_slug="notion", crawl=False)
        assert should_crawl(server) is False

    def test_disabled_and_system_servers_never_crawl(self):
        assert should_crawl(_server(name="notion", catalog_slug="notion", enabled=False)) is False
        assert should_crawl(_server(name="notion", catalog_slug="notion", system=True)) is False


class TestToolMatching:
    def test_exact_name_wins(self):
        assert resolve_tool(["search", "notion-search"], ["search"]) == "search"

    def test_prefixed_name_matches_the_plain_candidate(self):
        # The same server ships as `search`, `notion-search` and `notion_search`
        # depending on the packaging; a plan lists one and must find all three.
        assert resolve_tool(["notion-search"], ["search"]) == "notion-search"
        assert resolve_tool(["notion_search"], ["search"]) == "notion_search"

    def test_candidate_order_decides_between_matches(self):
        names = ["get_issues", "list_issues"]
        assert resolve_tool(names, ["list_issues", "get_issues"]) == "list_issues"

    def test_no_match_returns_none(self):
        assert resolve_tool(["get_weather"], ["list_issues"]) is None


class TestArgumentResolution:
    def test_argument_resolved_by_declared_name(self):
        schema = {"properties": {"boardId": {"type": "string"}}}
        assert resolve_argument(schema, ("board_id", "idBoard")) == "boardId"

    def test_single_required_argument_is_the_fallback(self):
        schema = {"properties": {"page": {"type": "string"}}, "required": ["page"]}
        assert resolve_argument(schema, ("id", "url")) == "page"

    def test_undeclared_arguments_are_dropped(self):
        schema = {"properties": {"query": {"type": "string"}}}
        assert filter_arguments(schema, {"limit": 50, "query": "x"}) == {"query": "x"}

    def test_schema_without_properties_passes_arguments_through(self):
        # An empty schema tells us nothing about the tool, which is not the
        # same as telling us it takes no arguments.
        assert filter_arguments({}, {"query": ""}) == {"query": ""}


class TestGenericPlan:
    def test_listing_tools_become_steps(self):
        server = _server(tools=[
            _tool("list_projects"),
            _tool("get_project", {"properties": {"id": {}}, "required": ["id"]}),
            _tool("create_project"),
        ])
        labels = [s.label for s in generic_plan(server).steps]
        assert labels == ["list_projects"]

    def test_search_tool_is_called_with_an_empty_query(self):
        server = _server(tools=[
            _tool("search", {"properties": {"query": {"type": "string"}}, "required": ["query"]}),
        ])
        step = generic_plan(server).steps[0]
        assert step.arguments == {"query": ""}

    def test_tool_needing_an_id_is_not_an_entry_point(self):
        server = _server(tools=[
            _tool("list_cards", {"properties": {"list_id": {}}, "required": ["list_id"]}),
        ])
        assert generic_plan(server).steps == ()

    def test_plural_getter_still_counts_as_a_listing(self):
        server = _server(tools=[_tool("get_boards")])
        assert [s.label for s in generic_plan(server).steps] == ["get_boards"]

    def test_step_count_is_capped(self):
        server = _server(tools=[_tool(f"list_thing{i}") for i in range(20)])
        assert len(generic_plan(server).steps) <= 6
