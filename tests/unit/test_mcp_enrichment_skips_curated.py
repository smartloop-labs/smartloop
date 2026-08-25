"""LLM enrichment must only run for hand-added MCP servers.

``_enrich_mcp_server`` exists for the ``uvx``/``npx`` case: a package that
ships no description and is not in our curated catalog. Servers installed
from the catalog arrive with ``description`` and ``tags`` already set, and
running enrichment on those burns two inference calls (plus possibly a model
load) only to overwrite curated copy with generated copy.
"""

import inspect
import threading
from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.model.mcp import MCPRegisterRequest
from smartloop.api.services.model_service import ModelService
from smartloop.workspace.workspace import MCPServer, MCPToolDefinition

_TOOLS = [MCPToolDefinition(name="get_weather", description="d", input_schema={"type": "object"})]


def _run_enrichment(**fields):
    """Run the real _enrich_mcp_server against a stub summarizer.

    Returns ``(server, calls)`` where ``calls`` records which summarizer
    methods — i.e. which LLM round-trips — actually fired.
    """
    server = MCPServer(id="x", name="demo", tools=_TOOLS, **fields)
    svc = MagicMock()
    svc._llm = MagicMock()
    svc._get_large_llm = MagicMock(return_value=MagicMock())
    svc._inference_lock = threading.Semaphore(1)

    calls = []
    summarizer = MagicMock()
    summarizer.generate_metadata.side_effect = lambda s: (
        calls.append("generate_metadata")
        or {"description": "GEN-DESC", "tags": ["gen1", "gen2"]}
    )
    summarizer.invoke.side_effect = lambda s: (calls.append("invoke") or "GEN-SUMMARY")

    with patch(
        "smartloop.agents.mcp.server_summarizer.MCPServerSummarizer",
        return_value=summarizer,
    ):
        ModelService._enrich_mcp_server(svc, server)
    return server, calls


class TestCuratedServersSkipEnrichment:
    def test_curated_description_and_tags_make_no_llm_calls(self):
        server, calls = _run_enrichment(description="Curated desc", tags=["weather", "api"])
        assert calls == []
        assert server.description == "Curated desc"
        assert server.tags == ["weather", "api"]

    def test_missing_summary_falls_back_to_description_for_free(self):
        """The catalog has no summary field, so the selector reuses the
        description rather than paying for a generated one."""
        server, calls = _run_enrichment(description="Curated desc", tags=["weather"])
        assert calls == []
        assert server.summary == "Curated desc"

    def test_curated_summary_is_left_alone(self):
        server, calls = _run_enrichment(
            description="Curated desc", tags=["weather"], summary="Curated summary"
        )
        assert calls == []
        assert server.summary == "Curated summary"


class TestHandAddedServersStillEnrich:
    def test_bare_uvx_add_generates_everything(self):
        server, calls = _run_enrichment()
        assert calls == ["generate_metadata", "invoke"]
        assert server.description == "GEN-DESC"
        assert server.tags == ["gen1", "gen2"]
        assert server.summary == "GEN-SUMMARY"

    def test_description_without_tags_still_enriches(self):
        """A package that advertises its own description but is not in our
        listings has no tags — it still needs them generated."""
        server, calls = _run_enrichment(description="Package's own desc")
        assert "generate_metadata" in calls
        assert server.description == "Package's own desc"  # not overwritten
        assert server.tags == ["gen1", "gen2"]

    def test_no_tools_discovered_skips_entirely(self):
        server = MCPServer(id="x", name="demo", tools=[])
        svc = MagicMock()
        with patch(
            "smartloop.agents.mcp.server_summarizer.MCPServerSummarizer"
        ) as summarizer_cls:
            ModelService._enrich_mcp_server(svc, server)
        summarizer_cls.assert_not_called()


class TestCatalogProvenanceSkipsEnrichment:
    """A resolved catalog slug is provenance the backend established itself, so
    it short-circuits enrichment even when the entry left a field blank."""

    def test_slug_alone_skips_enrichment(self):
        server, calls = _run_enrichment(catalog_slug="linear")
        assert calls == []

    def test_slug_skips_even_without_tags(self):
        """A curated entry with no tags must not fall through to the LLM."""
        server, calls = _run_enrichment(description="Curated desc", catalog_slug="linear")
        assert calls == []
        assert server.description == "Curated desc"

    def test_no_slug_and_nothing_curated_still_enriches(self):
        server, calls = _run_enrichment()
        assert calls == ["generate_metadata", "invoke"]
        assert server.catalog_slug is None


class TestCuratedMetadataCanReachTheServer:
    """The skip above is only reachable if the register API carries the slug
    through — without it the record always lands bare."""

    def test_register_request_accepts_catalog_slug(self):
        assert "catalog_slug" in MCPRegisterRequest.model_fields
        assert MCPRegisterRequest(server_url="https://x.test").catalog_slug is None

    @pytest.mark.parametrize(
        "method", ["add_local_mcp_server", "add_remote_mcp_server"]
    )
    @pytest.mark.parametrize("param", ["tags", "catalog_slug", "description"])
    def test_service_add_methods_accept_curated_fields(self, method, param):
        assert param in inspect.signature(getattr(ModelService, method)).parameters

    def test_server_record_persists_provenance(self):
        assert "catalog_slug" in MCPServer.model_fields
