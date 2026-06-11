"""
Regression: small models sometimes copy a parameter's schema description
text into the value (e.g. description := "Content as Markdown. Do not
escape the string ..."). Such echoes must be dropped, not sent to the tool.
"""
import json
from unittest.mock import MagicMock

from smartloop.agents.mcp.argument_extractor import MCPArgumentExtractor


SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Issue title"},
        "description": {
            "type": "string",
            "description": (
                "Content as Markdown. Do not escape the string — use literal "
                "newlines and special characters, not escape sequences."
            ),
        },
    },
    "required": ["title", "description"],
}


def _extractor(llm_response: str):
    llm = MagicMock()
    llm.get_num_tokens = MagicMock(return_value=10)
    llm.invoke = MagicMock(return_value=llm_response)
    return MCPArgumentExtractor(llm)


def test_echoed_description_is_dropped_and_flagged_missing():
    # LLM echoes the schema description as the description value.
    resp = json.dumps({
        "title": "find billing discrepancies",
        "description": (
            "Content as Markdown. Do not escape the string — use literal "
            "newlines and special characters, not escape sequences."
        ),
    })
    args = _extractor(resp).invoke(
        user_input="find billing discrepancies",
        tool_name="save_issue",
        input_schema=SCHEMA,
    )
    # The echoed description must not be passed through as a value.
    assert args.get("description") != SCHEMA["properties"]["description"]["description"]
    assert "description" not in args or args["description"] != ""
    # title is a real value and survives.
    assert args["title"] == "find billing discrepancies"
    # description is dropped → flagged missing.
    assert "description" in args.get("_missing_params", [])


def test_real_description_value_is_kept():
    resp = json.dumps({
        "title": "Billing mismatch",
        "description": "Invoice INV-42 charged $540 but PO approved $495.",
    })
    args = _extractor(resp).invoke(
        user_input="log the billing mismatch",
        tool_name="save_issue",
        input_schema=SCHEMA,
    )
    assert args["description"] == "Invoice INV-42 charged $540 but PO approved $495."
    assert "_missing_params" not in args


def test_is_schema_echo_helper():
    ex = MCPArgumentExtractor(MagicMock())
    info = SCHEMA["properties"]["description"]
    assert ex._is_schema_echo(info["description"], info) is True
    assert ex._is_schema_echo("a genuine markdown report", info) is False
    # short descriptions never trigger (avoid dropping legit short values)
    assert ex._is_schema_echo("DEV", {"description": "team"}) is False
