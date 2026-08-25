"""Crawl plan resolution and tool matching.

A plan only fires if it can find the right tool on the server, and only the
document-type servers should be crawled by default — otherwise a shell or
weather connection gets walked every refresh cycle for nothing.
"""

from types import SimpleNamespace

from smartloop.mcp.crawl_plans import (
    declares_argument,
    filter_arguments,
    generic_plan,
    resolve_argument,
    resolve_plan,
    resolve_tool,
    should_crawl,
)
from smartloop.mcp.plan_loader import plans as _plans

NOTION_PLAN = _plans()["notion"]
LINEAR_PLAN = _plans()["linear"]
TRELLO_PLAN = _plans()["trello"]


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
        assert resolve_plan(server).slug == "notion"

    def test_hand_added_server_matches_on_name(self):
        assert resolve_plan(_server(name="linear-mcp")).slug == "linear"

    def test_match_can_come_from_the_remote_url(self):
        server = _server(
            name="tasks",
            remote_config=SimpleNamespace(server_url="https://mcp.trello.com/mcp"),
        )
        assert resolve_plan(server).slug == "trello"

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


class TestCuratedPlanShape:
    """The curated plans are tree walks, so the tree has to hold together."""

    def _steps(self, plan):
        return {step.label: step for step in plan.steps}

    def test_every_parent_is_produced_by_an_earlier_step(self):
        # A step can only fan out over records another step already collected,
        # so a parent named later (or misspelled) silently never runs.
        for plan in (NOTION_PLAN, LINEAR_PLAN, TRELLO_PLAN):
            seen = set()
            for step in plan.steps:
                for parent in step.parent_labels:
                    assert parent in seen, f"{plan.slug}/{step.label} -> {parent}"
                seen.add(step.label)

    def test_step_labels_are_unique(self):
        for plan in (NOTION_PLAN, LINEAR_PLAN, TRELLO_PLAN):
            labels = [s.label for s in plan.steps]
            assert len(labels) == len(set(labels)), plan.slug

    def test_linear_reaches_issues_through_teams(self):
        steps = self._steps(LINEAR_PLAN)
        assert steps["teams"].parent_labels == ()
        assert steps["team-issues"].parent_labels == ("teams",)
        assert "list_issues" in steps["team-issues"].tools
        assert "teamId" in steps["team-issues"].parent_args

    def test_linear_comments_pool_every_issue_listing(self):
        comments = self._steps(LINEAR_PLAN)["issue-comments"]
        assert set(comments.parent_labels) == {
            "issues", "my-issues", "team-issues", "project-issues",
        }

    def test_notion_reaches_database_rows(self):
        pages = self._steps(NOTION_PLAN)["database-pages"]
        assert set(pages.parent_labels) == {"databases", "database-search"}
        assert pages.detail_tools, "a database row still needs its page body"

    def test_notion_covers_both_tool_vocabularies(self):
        # The hosted server and the OpenAPI package name nothing alike, and the
        # loose matcher pairs on affixes — `retrieve-a-page` matches neither
        # `retrieve_page` nor `get_page`, so both spellings are listed.
        search = self._steps(NOTION_PLAN)["pages"]
        assert resolve_tool(["notion-search"], search.tools) == "notion-search"
        assert resolve_tool(["API-post-search"], search.tools) == "API-post-search"
        assert resolve_tool(["notion-fetch"], search.detail_tools) == "notion-fetch"
        assert resolve_tool(
            ["API-get-block-children"], search.detail_tools
        ) == "API-get-block-children"

    def test_a_step_that_only_exists_for_one_argument_declares_it(self):
        # `database-search` is the `pages` search narrowed to databases; with
        # the filter dropped it would be a duplicate call.
        assert self._steps(NOTION_PLAN)["database-search"].needs_arguments == ("filter",)

    def test_trello_walks_the_hosted_action_dispatch_tools(self):
        # mcp.trello.com ships one tool per entity and dispatches on `action`,
        # so the walk is expressed in arguments rather than in tool names.  The
        # plan shipped before this was written for a package's `list_boards` /
        # `get_lists_by_board_id` naming, matched nothing, and reported a clean
        # zero-record crawl.
        hosted = [
            "trelloReadWorkspace", "trelloReadBoard", "trelloReadList",
            "trelloReadCard", "trelloReadInbox", "trelloReadMember",
        ]
        steps = self._steps(TRELLO_PLAN)
        assert resolve_tool(hosted, steps["boards"].tools) == "trelloReadBoard"
        assert resolve_tool(hosted, steps["lists"].tools) == "trelloReadList"
        assert resolve_tool(hosted, steps["cards"].tools) == "trelloReadCard"
        assert steps["boards"].arguments["action"] == "list"
        assert steps["lists"].arguments["action"] == "list_by_board"
        assert steps["cards"].arguments["action"] == "list_by_list"

    def test_trello_reaches_cards_through_boards_and_lists(self):
        steps = self._steps(TRELLO_PLAN)
        assert set(steps["lists"].parent_labels) == {"boards", "workspace-boards"}
        assert steps["cards"].parent_labels == ("lists",)
        assert "boardId" in steps["lists"].parent_args
        assert "listId" in steps["cards"].parent_args

    def test_trello_still_matches_a_packaged_server(self):
        # The snake_case candidates are the self-hosted packages' convention.
        # They cost nothing when absent, and a plan that only spoke to the
        # hosted server would drop those users.
        steps = self._steps(TRELLO_PLAN)
        assert resolve_tool(["list_boards"], steps["boards"].tools) == "list_boards"
        assert resolve_tool(
            ["get_cards_by_list_id"], steps["cards"].tools
        ) == "get_cards_by_list_id"

    def test_trello_does_not_re_fetch_a_card_the_hosted_row_already_holds(self):
        # The hosted card row carries description, labels, members, checklists
        # and the list/board names — its `get` returns the same object.  Listing
        # `trelloReadCard` as a detail tool would resolve there and spend one
        # call per card to learn nothing, so only the packaged fetch is named.
        cards = self._steps(TRELLO_PLAN)["cards"]
        assert resolve_tool(["trelloReadCard"], cards.detail_tools) is None
        assert resolve_tool(["get_card"], cards.detail_tools) == "get_card"

    def test_every_trello_step_says_which_action_it_wants(self):
        # `action` is required on every hosted tool, and which *other* arguments
        # are legal depends on its value — so a step that forgets it is not a
        # reduced call, it is a rejected one.
        for step in TRELLO_PLAN.steps:
            assert "action" in step.arguments, step.label


class TestNeedsArguments:
    def test_declared_argument_lets_the_step_run(self):
        schema = {"properties": {"query": {}, "filter": {}}}
        assert declares_argument(schema, ("filter",)) is True

    def test_undeclared_argument_stops_the_step(self):
        assert declares_argument({"properties": {"query": {}}}, ("filter",)) is False

    def test_an_empty_schema_is_not_evidence_against_the_step(self):
        assert declares_argument({}, ("filter",)) is True


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

    def test_a_tool_needing_a_parent_id_becomes_a_child_step(self):
        # The half of a server that is only reachable per-parent: without this
        # `list_issues` is dropped for requiring an argument, and every issue
        # on the server stays invisible.
        server = _server(tools=[
            _tool("list_teams"),
            _tool("list_issues", {"properties": {"teamId": {}}, "required": ["teamId"]}),
        ])
        steps = {s.label: s for s in generic_plan(server).steps}
        assert steps["list_issues"].parent_labels == ("list_teams",)
        assert "teamId" in steps["list_issues"].parent_args

    def test_a_child_step_comes_after_the_parent_it_walks(self):
        server = _server(tools=[
            _tool("list_cards", {"properties": {"board_id": {}}, "required": ["board_id"]}),
            _tool("list_boards"),
        ])
        labels = [s.label for s in generic_plan(server).steps]
        assert labels == ["list_boards", "list_cards"]

    def test_a_bare_id_names_no_parent_so_it_stays_dropped(self):
        # `id` says nothing about which listing could supply it.
        server = _server(tools=[
            _tool("list_teams"),
            _tool("list_items", {"properties": {"id": {}}, "required": ["id"]}),
        ])
        assert [s.label for s in generic_plan(server).steps] == ["list_teams"]

    def test_a_child_with_no_matching_parent_listing_is_dropped(self):
        server = _server(tools=[
            _tool("list_teams"),
            _tool("list_cards", {"properties": {"board_id": {}}, "required": ["board_id"]}),
        ])
        assert [s.label for s in generic_plan(server).steps] == ["list_teams"]
