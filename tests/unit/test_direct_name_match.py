"""Tests for MCP explorer direct (name-invocation) skill matching.

``match_skills`` unions an LLM qualifier (intent-aware) with a deterministic
*direct* match that fires only when the user names the skill explicitly.
These tests pin the direct matcher: a bare topic mention must NOT match — that
judgement is left to the LLM — while an explicit invocation does.
"""

from smartloop.agents.mcp.explorer import Explorer


class TestDirectNameMatches:
    FUND_SKILL = {
        "name": "fund-info",
        "description": (
            "Use this skill when user requests to summarize the funds, like "
            "fund names, ticker symbol, annual fund operating expenses"
        ),
        "content": "Extract fund names, ticker symbols and operating expenses.",
    }

    def test_topic_mention_is_not_a_direct_match(self):
        """A general question that only mentions the subject must NOT match."""
        result = Explorer._direct_name_matches(
            "what is the purpose of the funds ?", [self.FUND_SKILL]
        )
        assert result == []

    def test_summarize_request_is_not_a_direct_match(self):
        """'summarize the fund' is left to the LLM, not the direct matcher."""
        result = Explorer._direct_name_matches(
            "summarize the fund", [self.FUND_SKILL]
        )
        assert result == []

    def test_explicit_invocation_matches(self):
        """Naming the skill's phrase directly is a match."""
        result = Explorer._direct_name_matches(
            "give me the fund info for my portfolio", [self.FUND_SKILL]
        )
        assert [s["name"] for s in result] == ["fund-info"]

    def test_hyphenated_skill_name_matches(self):
        """The hyphenated name form is matched too."""
        result = Explorer._direct_name_matches(
            "run the fund-info skill", [self.FUND_SKILL]
        )
        assert [s["name"] for s in result] == ["fund-info"]

    def test_partial_word_does_not_match(self):
        """A substring that isn't a whole-word phrase must NOT match."""
        result = Explorer._direct_name_matches(
            "refunding information request", [self.FUND_SKILL]
        )
        assert result == []
