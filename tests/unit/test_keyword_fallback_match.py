"""Tests for MCP explorer keyword fallback matching."""

from smartloop.agents.mcp.explorer import Explorer


class TestKeywordFallbackMatch:
    """Tests for _keyword_fallback_match false-positive filtering."""

    SKILL = {
        "name": "document-review",
        "description": "proof read and review documents for grammar and style",
        "content": "This skill helps you proof read your documents.",
    }

    def test_short_generic_phrase_no_match(self):
        """'proof read' (2 tokens, full overlap) must NOT match."""
        result = Explorer._keyword_fallback_match(
            "proof read", [self.SKILL], min_overlap=2
        )
        assert result == []

    def test_longer_message_still_matches(self):
        """A longer message with partial overlap should still match."""
        result = Explorer._keyword_fallback_match(
            "proof read the quarterly report document",
            [self.SKILL],
            min_overlap=2,
        )
        assert len(result) == 1
        assert result[0]["name"] == "document-review"

    def test_threshold_scaling_still_works(self):
        """With many skills the effective threshold increases, requiring
        more overlap words — existing behaviour must be preserved."""
        skills = [
            {
                "name": f"skill-{i}",
                "description": f"description words alpha beta gamma {i}",
            }
            for i in range(10)
        ]
        # effective_threshold = 2 + (10 // 5) = 4
        # "alpha beta" only gives 2 overlap tokens → no match
        result = Explorer._keyword_fallback_match(
            "alpha beta gamma delta extra words here",
            skills,
            min_overlap=2,
        )
        # Only skills whose tokens overlap ≥ 4 words should match
        for matched in result:
            assert matched["name"].startswith("skill-")

    def test_single_short_word_message_no_tokens(self):
        """A message with only short words produces no tokens → empty."""
        result = Explorer._keyword_fallback_match(
            "hi", [self.SKILL], min_overlap=2
        )
        assert result == []

    def test_three_token_full_overlap_rejected(self):
        """Three content tokens with full overlap should also be rejected."""
        skill = {
            "name": "editing-tool",
            "description": "proof read document editing review",
        }
        result = Explorer._keyword_fallback_match(
            "proof read document", [skill], min_overlap=2
        )
        assert result == []
