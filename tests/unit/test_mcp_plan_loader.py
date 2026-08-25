"""Crawl plans are data, loaded from TOML.

They used to be Python literals, which made adding one a code change and
shipping one a release. What matters here: the bundled file still produces the
plans the crawler expects, a user can drop in their own without touching the
built-ins, and a malformed file is survivable — one bad drop-in must not take
the Notion connection down with it.
"""

import os

import pytest

from smartloop.mcp.crawl_plans import CrawlPlan, CrawlStep
from smartloop.mcp.plan_loader import (
    BUNDLED_PLANS_PATH,
    parse_definitions,
    load_plans,
    parse_plans,
    plans,
    reload_plans,
    user_plan_paths,
)

_MINIMAL = '''
[[plan]]
slug = "acme"
match = ["acme"]

[[plan.step]]
label = "widgets"
tools = ["list_widgets"]
'''


class TestBundledPlans:
    def test_the_shipped_plans_load(self):
        registry = load_plans(home_dir="/nonexistent")
        assert set(registry) == {"notion", "linear", "trello", "canva"}

    def test_linear_still_walks_teams_to_issues(self):
        # The tree the plan exists for. A transcription slip in the TOML would
        # show up here rather than as a quietly smaller index.
        steps = {s.label: s for s in load_plans(home_dir="/nonexistent")["linear"].steps}
        assert steps["team-issues"].parent_labels == ("teams",)
        assert "teamId" in steps["team-issues"].parent_args
        assert set(steps["issue-comments"].parent_labels) == {
            "issues", "my-issues", "team-issues", "project-issues",
        }

    def test_aliases_expand_in_place(self):
        # `detail_tools = ["@notion_fetch"]` has to become the eleven real
        # names, in order — the crawler matches candidates first-wins.
        pages = {s.label: s for s in load_plans(home_dir="/nonexistent")["notion"].steps}["pages"]
        assert pages.detail_tools[0] == "notion-fetch"
        assert "API-get-block-children" in pages.detail_tools
        assert len(pages.detail_tools) > 5

    def test_the_bundled_file_ships_with_the_package(self):
        assert os.path.isfile(BUNDLED_PLANS_PATH)


class TestParsing:
    def test_a_minimal_plan_parses(self):
        (plan, extend), = parse_plans(_MINIMAL, "test")
        assert plan.slug == "acme"
        assert extend is False
        assert [s.label for s in plan.steps] == ["widgets"]

    def test_an_alias_can_be_mixed_with_literal_names(self):
        text = '''
[alias]
common = ["a", "b"]

[[plan]]
slug = "acme"

[[plan.step]]
label = "x"
tools = ["@common", "c"]
'''
        (plan, _), = parse_plans(text, "test")
        assert plan.steps[0].tools == ("a", "b", "c")

    def test_a_named_fan_out_parses_and_expands_its_alias(self):
        # A watchlist is exactly the kind of list worth naming once and reusing
        # across every step that needs it.
        text = """
[alias]
watchlist = ["AAPL", "MSFT"]

[[plan]]
slug = "yahoo-finance"

[[plan.step]]
label = "profile"
tools = ["get_stock_info"]
for_each = { ticker = ["@watchlist"] }
"""
        (plan, _), = parse_plans(text, "test")
        assert plan.steps[0].for_each == {"ticker": ("AAPL", "MSFT")}

    def test_a_step_cannot_fan_out_both_ways(self):
        text = """
[[plan]]
slug = "x"

[[plan.step]]
label = "root"
tools = ["t"]

[[plan.step]]
label = "both"
tools = ["t"]
parent = "root"
for_each = { ticker = ["AAPL"] }
"""
        with pytest.raises(Exception, match="for_each"):
            parse_plans(text, "test")

    def test_an_unknown_alias_is_an_error_naming_it(self):
        text = _MINIMAL.replace('tools = ["list_widgets"]', 'tools = ["@nope"]')
        with pytest.raises(ValueError, match="@nope"):
            parse_plans(text, "test")

    def test_a_misspelled_field_is_rejected(self):
        # Silently ignoring it would leave a step that looks configured and
        # isn't.
        text = _MINIMAL + "\nmax_recordz = 10\n"
        with pytest.raises(Exception):
            parse_plans(text, "test")

    def test_a_parent_that_is_not_an_earlier_step_is_rejected(self):
        text = '''
[[plan]]
slug = "acme"

[[plan.step]]
label = "x"
tools = ["t"]
parent = "nowhere"
'''
        with pytest.raises(Exception, match="nowhere"):
            parse_plans(text, "test")

    def test_a_parent_declared_later_is_rejected(self):
        # It would never run: a step only fans out over records an earlier step
        # already collected.
        text = '''
[[plan]]
slug = "acme"

[[plan.step]]
label = "child"
tools = ["t"]
parent = "root"

[[plan.step]]
label = "root"
tools = ["t"]
'''
        with pytest.raises(Exception, match="root"):
            parse_plans(text, "test")

    def test_duplicate_labels_are_rejected(self):
        text = _MINIMAL + '''
[[plan.step]]
label = "widgets"
tools = ["other"]
'''
        with pytest.raises(Exception, match="duplicate"):
            parse_plans(text, "test")


class TestUserPlans:
    def _write(self, home, name, text, in_dropin_dir=True):
        directory = os.path.join(home, "mcp.d") if in_dropin_dir else home
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def test_a_dropped_in_plan_joins_the_built_ins(self, tmp_path):
        self._write(str(tmp_path), "acme.toml", _MINIMAL)
        registry = load_plans(home_dir=str(tmp_path))
        assert "acme" in registry
        assert "notion" in registry

    def test_a_single_user_file_also_works(self, tmp_path):
        self._write(str(tmp_path), "mcp.toml", _MINIMAL, in_dropin_dir=False)
        assert "acme" in load_plans(home_dir=str(tmp_path))

    def test_a_matching_slug_replaces_the_built_in(self, tmp_path):
        override = _MINIMAL.replace('slug = "acme"', 'slug = "notion"')
        self._write(str(tmp_path), "notion.toml", override)
        notion = load_plans(home_dir=str(tmp_path))["notion"]
        assert [s.label for s in notion.steps] == ["widgets"]

    def test_extend_appends_to_the_built_in(self, tmp_path):
        # The point of `extend`: add one step without restating the other ten.
        self._write(str(tmp_path), "notion-extra.toml", '''
[[plan]]
slug = "notion"
extend = true

[[plan.step]]
label = "my-extra-listing"
tools = ["list_my_things"]
''')
        notion = load_plans(home_dir=str(tmp_path))["notion"]
        labels = [s.label for s in notion.steps]
        assert labels[-1] == "my-extra-listing"
        assert "recent-pages" in labels

    def test_an_extension_can_fan_out_from_a_built_in_step(self, tmp_path):
        self._write(str(tmp_path), "notion-extra.toml", '''
[[plan]]
slug = "notion"
extend = true

[[plan.step]]
label = "page-attachments"
tools = ["list_attachments"]
parent = "recent-pages"
''')
        notion = load_plans(home_dir=str(tmp_path))["notion"]
        step = next(s for s in notion.steps if s.label == "page-attachments")
        assert step.parent_labels == ("recent-pages",)

    def test_an_extension_colliding_with_a_built_in_label_is_rejected(self, tmp_path):
        self._write(str(tmp_path), "bad.toml", '''
[[plan]]
slug = "notion"
extend = true

[[plan.step]]
label = "users"
tools = ["list_users"]
''')
        # Rejected, and the built-in plan survives intact.
        notion = load_plans(home_dir=str(tmp_path))["notion"]
        assert [s.label for s in notion.steps].count("users") == 1

    def test_a_malformed_drop_in_does_not_take_the_others_down(self, tmp_path):
        self._write(str(tmp_path), "aaa-broken.toml", "this is not [valid toml")
        self._write(str(tmp_path), "bbb-good.toml", _MINIMAL)
        registry = load_plans(home_dir=str(tmp_path))
        assert "acme" in registry
        assert "notion" in registry

    def test_drop_ins_are_read_in_name_order(self, tmp_path):
        self._write(str(tmp_path), "10-first.toml", _MINIMAL)
        self._write(str(tmp_path), "20-second.toml", _MINIMAL.replace(
            'label = "widgets"', 'label = "gadgets"',
        ))
        registry = load_plans(home_dir=str(tmp_path))
        # Later file wins for the same slug.
        assert [s.label for s in registry["acme"].steps] == ["gadgets"]

    def test_user_paths_list_the_single_file_before_the_directory(self, tmp_path):
        self._write(str(tmp_path), "mcp.toml", _MINIMAL, in_dropin_dir=False)
        self._write(str(tmp_path), "z.toml", _MINIMAL)
        paths = user_plan_paths(str(tmp_path))
        assert paths[0].endswith("mcp.toml")
        assert paths[1].endswith(os.path.join("mcp.d", "z.toml"))

    def test_no_user_files_is_not_an_error(self, tmp_path):
        assert user_plan_paths(str(tmp_path)) == []


class TestCache:
    def test_the_registry_is_cached_and_reloadable(self):
        first = plans()
        assert plans() is first
        assert set(reload_plans()) == set(first)


class TestModelsAreStillUsableDirectly:
    """The models are the contract the TOML is validated against, so building
    one in Python — which the derived/generic plan does — still has to work."""

    def test_a_plan_can_be_constructed_in_code(self):
        plan = CrawlPlan(
            slug="x",
            steps=(CrawlStep(label="a", tools=("t",)),),
            derived=True,
        )
        assert plan.steps[0].max_records == 40
        assert plan.derived is True

    def test_a_step_without_tools_is_rejected(self):
        with pytest.raises(Exception):
            CrawlStep(label="a", tools=())


class TestServerDefinitions:
    """A file describes the server, not just how to index it — which is what
    makes a dropped-in file enough to add a connection."""

    def test_a_local_server_carries_its_command(self):
        text = """
[[mcp]]
name = "yahoo-finance"
package = "yahoo-finance-mcp"
"""
        d = parse_definitions(text, "test")["yahoo-finance"]
        assert d.launch == ("uvx", ["yahoo-finance-mcp"])
        assert d.package == "yahoo-finance-mcp"
        assert d.kind == "local"

    def test_a_remote_server_carries_its_url(self):
        text = """
[[mcp]]
name = "my-wiki"
url = "https://mcp.example.com/mcp"
"""
        d = parse_definitions(text, "test")["my-wiki"]
        assert d.url == "https://mcp.example.com/mcp"
        assert d.kind == "remote"

    def test_indexing_is_optional(self):
        # Most servers have no content to enumerate; connecting is the whole
        # point, and the tools get indexed either way.
        text = """
[[mcp]]
name = "weather"
package = "weather-mcp"
"""
        d = parse_definitions(text, "test")["weather"]
        assert d.index == ()
        assert d.plan.steps == ()

    def test_a_package_name_becomes_a_uvx_launch(self):
        # The same thing typing it into "Add your own" does, so a file and the
        # field install the same server rather than two subtly different ones.
        text = """
[[mcp]]
name = "yahoo-finance"
package = "yahoo-finance-mcp"
"""
        d = parse_definitions(text, "test")["yahoo-finance"]
        assert d.kind == "local"
        assert d.launch == ("uvx", ["yahoo-finance-mcp"])

    def test_a_command_is_there_for_what_a_package_name_cannot_say(self):
        text = """
[[mcp]]
name = "custom"
command = "npx"
args = ["-y", "some-mcp"]
"""
        d = parse_definitions(text, "test")["custom"]
        assert d.launch == ("npx", ["-y", "some-mcp"])

    def test_a_package_and_a_command_say_the_same_thing_twice(self):
        text = """
[[mcp]]
name = "confused"
package = "x"
command = "npx"
"""
        with pytest.raises(Exception, match="twice"):
            parse_definitions(text, "test")

    def test_a_server_is_local_or_remote_not_both(self):
        text = """
[[mcp]]
name = "confused"
package = "x"
url = "https://x"
"""
        with pytest.raises(Exception, match="alternatives"):
            parse_definitions(text, "test")

    def test_the_name_becomes_the_key_and_a_match(self):
        text = """
[[mcp]]
name = "Yahoo-Finance"
match = ["yfinance"]
package = "yahoo-finance-mcp"
"""
        d = parse_definitions(text, "test")["yahoo-finance"]
        assert d.key == "yahoo-finance"
        assert "yfinance" in d.plan.match

    def test_a_definition_with_index_steps_still_produces_a_plan(self, tmp_path):
        path = tmp_path / "mcp.d"
        path.mkdir()
        (path / "yf.toml").write_text("""
[alias]
watchlist = ["AAPL", "MSFT"]

[[mcp]]
name = "yahoo-finance"
package = "yahoo-finance-mcp"

[[mcp.index]]
label = "profile"
tools = ["get_stock_info"]
for_each = { ticker = ["@watchlist"] }
""")
        registry = load_plans(home_dir=str(tmp_path))
        plan = registry["yahoo-finance"]
        assert [s.label for s in plan.steps] == ["profile"]
        assert plan.steps[0].for_each == {"ticker": ("AAPL", "MSFT")}


class TestIndexInterval:
    """How often a connection re-indexes belongs to the source, not the app."""

    def _parse(self, every):
        text = f"""
[[mcp]]
name = "air-quality"
package = "mcp-open-meteo"
index_every = {every}
"""
        return parse_definitions(text, "test")["air-quality"]

    def test_durations_carry_their_unit(self):
        assert self._parse('"30s"').index_interval == 30
        assert self._parse('"30m"').index_interval == 1800
        assert self._parse('"6h"').index_interval == 21600
        assert self._parse('"1d"').index_interval == 86400

    def test_a_bare_number_is_seconds(self):
        assert self._parse("900").index_interval == 900

    def test_unset_follows_the_app_wide_schedule(self):
        text = """
[[mcp]]
name = "wiki"
url = "https://mcp.example.com/mcp"
"""
        assert parse_definitions(text, "test")["wiki"].index_interval is None

    def test_a_typo_is_an_error_not_a_plan_that_never_refreshes(self):
        with pytest.raises(Exception, match="duration"):
            self._parse('"6 fortnights"')
