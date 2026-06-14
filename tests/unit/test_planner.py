"""Unit tests for the Planner module (classify → plan pipeline)."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from smartloop.agents.planner import (
    Planner,
    PlanContext,
    PlanStep,
    ExecutionPlan,
    StepType,
    MCPToolContext,
    IntentType,
    ClassificationResult,
    MatchedTool,
    SYSTEM_TOOLS_SERVER_ID,
    SYSTEM_TOOLS_SERVER_NAME,
    _extract_paths,
)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke = MagicMock(return_value="[]")
    llm.get_max_tokens = MagicMock(return_value=4096)
    llm.get_num_tokens = MagicMock(return_value=100)
    return llm


@pytest.fixture
def planner(mock_llm):
    return Planner(mock_llm)


@pytest.fixture
def sample_servers():
    return [
        {
            "id": "srv-1",
            "name": "Fetch Server",
            "tools": [
                {"name": "fetch", "description": "Fetch a URL and return content"},
            ],
        },
        {
            "id": "srv-2",
            "name": "GitHub",
            "tools": [
                {"name": "create-issue", "description": "Create a GitHub issue"},
            ],
        },
    ]


@pytest.fixture
def sample_mcp_tools():
    return [
        MCPToolContext(
            server_id="srv-1",
            server_name="Fetch Server",
            tools=[{"name": "fetch", "description": "Fetch a URL and return content"}],
        ),
        MCPToolContext(
            server_id="srv-2",
            server_name="GitHub",
            tools=[{"name": "create-issue", "description": "Create a GitHub issue"}],
        ),
    ]


# ------------------------------------------------------------------ #
# 1. build_context                                                     #
# ------------------------------------------------------------------ #

def test_build_context(planner, sample_servers):
    ctx = planner.build_context(
        user_message="Hello world",
        available_servers=sample_servers,
        cwd="/tmp",
        skills=["Be polite"],
        document_context="Some docs",
    )
    assert isinstance(ctx, PlanContext)
    assert ctx.user_message == "Hello world"
    assert ctx.skills == ["Be polite"]
    assert ctx.document_context == "Some docs"
    assert ctx.cwd == "/tmp"
    assert len(ctx.mcp_tools) == 2
    assert ctx.mcp_tools[0].server_id == "srv-1"
    assert ctx.mcp_tools[0].server_name == "Fetch Server"


def test_build_context_filters_system_tools(planner):
    servers = [
        {"id": "__system_tools__", "name": "smartloop", "tools": [{"name": "explore"}]},
        {"id": "ext-1", "name": "External", "tools": [{"name": "foo"}]},
    ]
    ctx = planner.build_context("test", servers)
    assert len(ctx.mcp_tools) == 1
    assert ctx.mcp_tools[0].server_id == "ext-1"


def test_build_context_detects_urls(planner):
    ctx = planner.build_context(
        user_message="Fetch https://example.com/page and summarize",
        available_servers=[],
    )
    assert "https://example.com/page" in ctx.detected_urls


# ------------------------------------------------------------------ #
# 2. classify — deterministic (Tier 1)                                #
# ------------------------------------------------------------------ #

def test_classify_conversation(planner):
    """No signals → LLM fallback → default CONVERSATION."""
    result = planner.classify("hello", mcp_tools=[])
    assert result.intent == IntentType.CONVERSATION


def test_classify_tool_call_mcp_keyword(planner, sample_mcp_tools):
    """MCP tool keyword match → TOOL_CALL."""
    result = planner.classify(
        "create an issue on GitHub",
        mcp_tools=sample_mcp_tools,
    )
    assert result.intent == IntentType.TOOL_CALL
    assert len(result.matched_tools) >= 1
    assert result.matched_tools[0].tool_name == "create-issue"


def test_classify_url_no_signal(planner, sample_mcp_tools, mock_llm):
    """URL alone is not a signal — falls through to LLM classifier.
    Explore step handles fetch internally."""
    mock_llm.invoke.return_value = "CONVERSATION"
    result = planner.classify(
        "Summarize https://example.com/article",
        mcp_tools=sample_mcp_tools,
    )
    assert "https://example.com/article" in result.detected_urls


def test_classify_action_bash(planner):
    """Bash block → ACTION(bash)."""
    result = planner.classify(
        'Run this:\n```bash\necho hello\n```',
        mcp_tools=[],
    )
    assert result.intent == IntentType.ACTION
    assert result.action_kind == "bash"
    assert "echo hello" in result.bash_commands


def test_classify_action_read(planner, tmp_path):
    """File path → ACTION(read)."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    result = planner.classify(
        f"Read {test_file}",
        mcp_tools=[],
        cwd=str(tmp_path),
    )
    assert result.intent == IntentType.ACTION
    assert result.action_kind == "read"


def test_classify_explore_directory(planner, tmp_path):
    """Directory path → EXPLORE."""
    sub = tmp_path / "src"
    sub.mkdir()
    result = planner.classify(
        f"Explore {sub}",
        mcp_tools=[],
        cwd=str(tmp_path),
    )
    assert result.intent == IntentType.EXPLORE


def test_classify_action_grep(planner):
    """Grep keywords → ACTION(grep)."""
    result = planner.classify(
        "search for TODO in the codebase",
        mcp_tools=[],
    )
    assert result.intent == IntentType.ACTION
    assert result.action_kind == "grep"


def test_classify_document_context(planner):
    """Document context available + no action signals → EXPLORE with document flag."""
    result = planner.classify(
        "What does the policy say about vacation days?",
        mcp_tools=[],
        document_context="Company policy: employees get 20 vacation days per year.",
    )
    assert result.intent == IntentType.EXPLORE
    assert result.has_document_context


def test_classify_document_context_with_tool_signal(planner, sample_mcp_tools):
    """Document context + MCP tool match → TOOL_CALL (URL not a signal)."""
    result = planner.classify(
        "Fetch https://example.com and compare to our docs",
        mcp_tools=sample_mcp_tools,
        document_context="Some relevant document content here.",
    )
    assert result.intent == IntentType.TOOL_CALL
    assert result.has_document_context


def test_classify_multi_step_write_intent(planner, sample_mcp_tools):
    """Write intent alone → MULTI_STEP (URL not a signal, explore handles fetch)."""
    result = planner.classify(
        "Fetch https://example.com and save it to output.md",
        mcp_tools=sample_mcp_tools,
    )
    assert result.has_write_intent
    assert len(result.detected_urls) >= 1


def test_classify_multi_step_write_only(planner):
    """Write intent alone (no paths/URLs) → MULTI_STEP."""
    result = planner.classify(
        "Write a file with a summary of the project to output.md",
        mcp_tools=[],
    )
    assert result.intent == IntentType.MULTI_STEP
    assert result.has_write_intent


def test_classify_llm_fallback(planner, mock_llm):
    """No deterministic signals → LLM classify called."""
    mock_llm.invoke.return_value = "TOOL_CALL"
    result = planner.classify(
        "do the thing with the API",
        mcp_tools=[],
        skills=["Use the API server for API calls"],
    )
    assert result.intent == IntentType.TOOL_CALL
    assert mock_llm.invoke.called


def test_classify_llm_fallback_unparseable(planner, mock_llm):
    """LLM returns garbage → default to CONVERSATION."""
    mock_llm.invoke.return_value = "I don't know what to do"
    result = planner.classify(
        "something ambiguous",
        mcp_tools=[],
    )
    assert result.intent == IntentType.CONVERSATION


# ------------------------------------------------------------------ #
# 3. create_plan with ClassificationResult                            #
# ------------------------------------------------------------------ #

def test_create_plan_conversation(planner):
    classification = ClassificationResult(intent=IntentType.CONVERSATION)
    ctx = PlanContext(user_message="hello")
    plan = planner.create_plan(classification, ctx)
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) == 0
    assert plan.reasoning == "conversation"


def test_create_plan_tool_call(planner, sample_servers):
    classification = ClassificationResult(
        intent=IntentType.TOOL_CALL,
        matched_tools=[MatchedTool(
            server_id="srv-2", server_name="GitHub", tool_name="create-issue",
        )],
    )
    ctx = planner.build_context("create issue", sample_servers)
    plan = planner.create_plan(classification, ctx)
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == StepType.TOOL
    assert plan.steps[0].tool_name == "create-issue"
    assert plan.steps[0].server_id == "srv-2"


def test_create_plan_tool_call_url_fetch(planner, sample_servers):
    classification = ClassificationResult(
        intent=IntentType.TOOL_CALL,
        detected_urls=["https://example.com/page"],
    )
    ctx = planner.build_context(
        "Fetch https://example.com/page",
        sample_servers,
    )
    plan = planner.create_plan(classification, ctx)
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == StepType.TOOL
    assert "fetch" in (plan.steps[0].tool_name or "").lower()


def test_create_plan_action_bash(planner):
    classification = ClassificationResult(
        intent=IntentType.ACTION,
        action_kind="bash",
        bash_commands=["echo hello"],
    )
    ctx = PlanContext(user_message='Run:\n```bash\necho hello\n```')
    plan = planner.create_plan(classification, ctx)
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == StepType.ACTION
    assert plan.steps[0].action_kind == "bash"
    assert "echo hello" in plan.steps[0].instruction


def test_create_plan_action_read(planner, tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    classification = ClassificationResult(
        intent=IntentType.ACTION,
        action_kind="read",
        detected_paths=[str(test_file)],
    )
    ctx = PlanContext(user_message=f"Read {test_file}", cwd=str(tmp_path))
    plan = planner.create_plan(classification, ctx)
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == StepType.ACTION
    assert plan.steps[0].action_kind == "read"


def test_create_plan_action_grep(planner):
    classification = ClassificationResult(
        intent=IntentType.ACTION,
        action_kind="grep",
        has_grep_intent=True,
    )
    ctx = PlanContext(user_message="search for TODO")
    plan = planner.create_plan(classification, ctx)
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == StepType.ACTION
    assert plan.steps[0].action_kind == "grep"


def test_create_plan_explore(planner, tmp_path):
    sub = tmp_path / "src"
    sub.mkdir()
    classification = ClassificationResult(
        intent=IntentType.EXPLORE,
        detected_paths=[str(sub)],
    )
    ctx = PlanContext(user_message=f"Explore {sub}", cwd=str(tmp_path))
    plan = planner.create_plan(classification, ctx)
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == StepType.EXPLORE


def test_create_plan_multi_step_from_signals(planner, sample_servers, mock_llm):
    """MULTI_STEP with LLM returning empty → falls back to signal-based steps."""
    mock_llm.invoke.return_value = "[]"
    classification = ClassificationResult(
        intent=IntentType.MULTI_STEP,
        detected_urls=["https://example.com"],
        has_write_intent=True,
    )
    ctx = planner.build_context(
        "Fetch https://example.com and save to output.md",
        sample_servers,
    )
    plan = planner.create_plan(classification, ctx)
    # Should have fetch + conversational + write from signal fallback
    assert len(plan.steps) >= 2
    types = [s.step_type for s in plan.steps]
    assert StepType.TOOL in types  # fetch


def test_create_plan_multi_step_from_llm(planner, mock_llm, sample_servers):
    """MULTI_STEP with LLM returning valid steps."""
    mock_llm.invoke.return_value = json.dumps([
        {
            "step": 1,
            "step_type": "tool",
            "tool_name": "fetch",
            "server_id": "srv-1",
            "server_name": "Fetch Server",
            "instruction": "Fetch the URL",
        },
        {
            "step": 2,
            "step_type": "action",
            "action_kind": "write",
            "tool_name": "file-write",
            "server_id": "__system_tools__",
            "server_name": "smartloop",
            "instruction": "Write to output.md",
        },
    ])
    classification = ClassificationResult(
        intent=IntentType.MULTI_STEP,
        detected_urls=["https://example.com"],
        has_write_intent=True,
    )
    ctx = planner.build_context(
        "Fetch https://example.com and save to output.md",
        sample_servers,
    )
    plan = planner.create_plan(classification, ctx)
    assert len(plan.steps) == 2
    assert plan.steps[0].step_type == StepType.TOOL
    assert plan.steps[1].step_type == StepType.ACTION


# ------------------------------------------------------------------ #
# 4. _parse_plan_response                                             #
# ------------------------------------------------------------------ #

def test_parse_plan_response_malformed(planner):
    plan = planner._parse_plan_response("not valid json", "test")
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) == 0
    assert "Parse error" in plan.reasoning


def test_parse_plan_response_not_list(planner):
    plan = planner._parse_plan_response('{"key": "value"}', "test")
    assert len(plan.steps) == 0
    assert "not a list" in plan.reasoning


def test_parse_plan_response_markdown_fences(planner):
    raw = '```json\n[{"step":1,"step_type":"explore","tool_name":"explore","instruction":"walk"}]\n```'
    plan = planner._parse_plan_response(raw, "test")
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == StepType.EXPLORE


def test_parse_plan_response_caps_at_5_steps(planner):
    steps_data = [
        {"step": i, "step_type": "conversational", "tool_name": "convo", "instruction": f"step {i}"}
        for i in range(1, 8)
    ]
    plan = planner._parse_plan_response(json.dumps(steps_data), "test")
    assert len(plan.steps) == 5


def test_backward_compat_tool_call_maps_to_tool(planner):
    plan = planner._parse_plan_response(json.dumps([
        {
            "step": 1, "step_type": "tool_call", "tool_name": "fetch",
            "server_id": "srv-1", "server_name": "Fetch", "instruction": "Fetch URL",
        }
    ]), "fetch")
    assert plan.steps[0].step_type == StepType.TOOL


def test_backward_compat_mcp_maps_to_tool(planner):
    plan = planner._parse_plan_response(json.dumps([
        {
            "step": 1, "step_type": "mcp", "tool_name": "fetch",
            "server_id": "srv-1", "server_name": "Fetch", "instruction": "Fetch URL",
        }
    ]), "fetch")
    assert plan.steps[0].step_type == StepType.TOOL


def test_action_step_type_corrected_to_tool_for_external_server(planner):
    plan = planner._parse_plan_response(json.dumps([
        {
            "step": 1, "step_type": "action", "tool_name": "fetch",
            "server_id": "srv-1", "server_name": "Fetch MCP Server",
            "instruction": "Fetch https://example.com and extract content.",
        }
    ]), "fetch")
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == StepType.TOOL
    assert plan.steps[0].server_id == "srv-1"


def test_action_kind_inferred_from_old_step_type(planner):
    plan = planner._parse_plan_response(json.dumps([
        {
            "step": 1, "step_type": "read", "tool_name": "file-read",
            "server_id": "__system_tools__", "server_name": "smartloop",
            "instruction": "Read main.py",
        }
    ]), "show me main.py")
    assert plan.steps[0].step_type == StepType.ACTION
    assert plan.steps[0].action_kind == "read"


# ------------------------------------------------------------------ #
# 5. _extract_paths                                                    #
# ------------------------------------------------------------------ #

def test_extract_paths(tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("x")
    found = _extract_paths(f"look at {test_file}", str(tmp_path))
    assert str(test_file) in found


def test_extract_paths_no_match(tmp_path):
    found = _extract_paths("no paths here", str(tmp_path))
    assert found == []
