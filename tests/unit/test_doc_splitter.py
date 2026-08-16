import json

import pytest

from smartloop.utils.doc_splitter import is_markdown_table, json_to_table


class TestJsonToTable:
    def test_converts_array_of_objects(self):
        data = [
            {"name": "Drake Maye", "team": "NE", "td": 31},
            {"name": "Jordan Love", "team": "GB", "td": 23},
        ]
        table = json_to_table(json.dumps(data))
        assert table is not None
        lines = table.splitlines()
        assert lines[0] == "| name | team | td |"
        assert lines[1] == "|---|---|---|"
        assert "| Drake Maye | NE | 31 |" in lines
        assert "| Jordan Love | GB | 23 |" in lines

    def test_accepts_rows_key(self):
        table = json_to_table(json.dumps({"category": "EPA/play", "rows": [{"name": "A", "v": 1}]}))
        assert table is not None
        assert "| name | v |" in table
        assert "| A | 1 |" in table

    def test_flattens_nested_objects_with_dot_keys(self):
        table = json_to_table(json.dumps({
            "rows": [{"name": "A", "values": {"epa": 0.296484, "td": 31}}]
        }))
        assert "| name | values.epa | values.td |" in table
        assert "| A | 0.2965 | 31 |" in table

    def test_rounds_floats_and_drops_integral_decimal(self):
        table = json_to_table(json.dumps([{"a": 0.1234567, "b": 3.0}]))
        assert "| 0.1235 | 3 |" in table

    def test_returns_none_for_non_tabular_json(self):
        assert json_to_table('{"hello": "world"}') is None
        assert json_to_table('[1, 2, 3]') is None
        assert json_to_table('[[1, 2], [3, 4]]') is None

    def test_returns_none_for_non_json(self):
        assert json_to_table("plain prose that is not json") is None
        assert json_to_table("") is None

    def test_caps_rows(self):
        rows = [{"i": i} for i in range(501)]
        table = json_to_table(json.dumps(rows))
        assert table is not None
        assert "(500 of 501 rows shown)" in table


class TestIsMarkdownTable:
    def test_detects_separator_row(self):
        assert is_markdown_table("| a | b |\n|---|---|\n| 1 | 2 |")

    def test_false_for_prose(self):
        assert not is_markdown_table("the quick brown fox jumps over the lazy dog")

    def test_false_for_data_row_without_separator(self):
        assert not is_markdown_table("| a | b |\n| 1 | 2 |")

    def test_false_for_empty(self):
        assert not is_markdown_table("")
