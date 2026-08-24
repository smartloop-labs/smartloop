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
        assert set(registry) == {"notion", "linear", "trello"}

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
