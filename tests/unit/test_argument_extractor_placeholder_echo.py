"""
Regression: small models sometimes emit a parameter's own name/description
as a placeholder value (e.g. path := "file_path", url := "your_url") instead
of a real value.  Such placeholders must be dropped so the param is treated
as missing — otherwise the tool runs with a bogus value (e.g. file-read tries
to open a file literally named "file_path").
"""
import json
from unittest.mock import MagicMock

from smartloop.agents.mcp.argument_extractor import MCPArgumentExtractor


FILE_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to read."},
        "query": {
            "type": "string",
            "description": "Optional search query to filter matching lines.",
        },
    },
    "required": ["path"],
}


def _extractor(llm_response: str):
    llm = MagicMock()
    llm.get_num_tokens = MagicMock(return_value=10)
    llm.invoke = MagicMock(return_value=llm_response)
    return MCPArgumentExtractor(llm)


def test_placeholder_path_is_dropped_and_flagged_missing():
    # LLM echoes the field name/description as the value ("what is my pitch").
    resp = json.dumps({"path": "file_path"})
    args = _extractor(resp).invoke(
        user_input="what is my pitch",
        tool_name="file-read",
        input_schema=FILE_READ_SCHEMA,
    )
    assert args.get("path") != "file_path"
    assert "path" in args.get("_missing_params", [])


def test_real_path_value_is_kept():
    resp = json.dumps({"path": "domain-specific-llms-proposal.md"})
    args = _extractor(resp).invoke(
        user_input="read the proposal file",
        tool_name="file-read",
        input_schema=FILE_READ_SCHEMA,
    )
    assert args["path"] == "domain-specific-llms-proposal.md"
    assert "_missing_params" not in args


def test_is_placeholder_echo_helper():
    ex = MCPArgumentExtractor(MagicMock())
    path_info = FILE_READ_SCHEMA["properties"]["path"]

    # Placeholders / name+description echoes are dropped.
    assert ex._is_placeholder_echo("path", "file_path", path_info) is True
    assert ex._is_placeholder_echo("path", "path", path_info) is True
    assert ex._is_placeholder_echo("url", "your_url", {"description": "The URL"}) is True
    assert ex._is_placeholder_echo("content", "string", {"description": "Content"}) is True

    # Real values survive: a filename with an extension, a path, a query term.
    assert ex._is_placeholder_echo("path", "about.md", path_info) is False
    assert ex._is_placeholder_echo("path", "src/app.py", path_info) is False
    assert ex._is_placeholder_echo("query", "pitch", FILE_READ_SCHEMA["properties"]["query"]) is False
    assert ex._is_placeholder_echo("symbol", "AAPL", {"description": "Stock ticker symbol"}) is False
