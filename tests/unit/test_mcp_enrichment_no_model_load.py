"""Registering an MCP server must never *load* a model.

``_enrich_mcp_server`` used to call ``_get_large_llm()``, which lazily loads
the largest downloaded model.  With ``SLP_LLM_LARGE_N_CTX_CAP`` at its default
of 0 the llama.cpp factory auto-sizes that model's context to swallow all
available memory — up to 131072 tokens on a >=32GB Mac, roughly 10GB of KV
cache — and nothing on the registration path ever unloads it
(``unload_large_model`` is only reached from the chat router).  So a single
``POST /v1/projects/{id}/mcp/register`` permanently parked ~10GB in the API
process in order to run two short summarization prompts over a tool list.

Enrichment now uses the small router model unconditionally — it never reads
``_large_llm`` at all.  The prompts are short and low-stakes (a one-line
description, three tags, a capability blurb), so they are not worth the large
model's memory even when a prior chat request already left it resident.
"""

import inspect
import threading
from unittest.mock import MagicMock, patch

from smartloop.api.services.model_service import ModelService
from smartloop.workspace.workspace import MCPServer, MCPToolDefinition

_TOOLS = [MCPToolDefinition(name="get_weather", description="d", input_schema={"type": "object"})]


def _run_enrichment(svc):
    """Run the real _enrich_mcp_server on *svc*, returning the LLM the
    summarizer was constructed with (or None if it never got that far)."""
    server = MCPServer(id="x", name="demo", tools=_TOOLS)
    svc._inference_lock = threading.Semaphore(1)

    got = {}
    summarizer_cls = MagicMock(
        side_effect=lambda llm: got.setdefault("llm", llm) and None or MagicMock()
    )
    with patch(
        "smartloop.agents.mcp.server_summarizer.MCPServerSummarizer", summarizer_cls
    ):
        ModelService._enrich_mcp_server(svc, server)
    return got.get("llm")


class TestEnrichmentNeverTriggersAModelLoad:
    def test_cold_process_uses_the_router_model_and_loads_nothing(self):
        """The regression test for the 11GB blowup: with no large model
        resident, enrichment must use the router model and must not ask for
        one to be loaded."""
        svc = MagicMock()
        svc._llm = MagicMock(name="router")
        svc._large_llm = None

        assert _run_enrichment(svc) is svc._llm
        svc._get_large_llm.assert_not_called()
        svc._detect_large_model.assert_not_called()

    def test_resident_large_model_is_still_not_used(self):
        """Even when a prior chat request left the large model resident,
        enrichment sticks to the router model — these prompts never justify the
        larger model's memory or its share of the inference slot."""
        svc = MagicMock()
        svc._llm = MagicMock(name="router")
        svc._large_llm = MagicMock(name="large")

        assert _run_enrichment(svc) is svc._llm
        svc._get_large_llm.assert_not_called()

    def test_router_model_is_loaded_when_absent(self):
        """The router model is the always-resident one; enrichment may load it
        (normally a no-op — lifespan preloads it at startup)."""
        svc = MagicMock()
        svc._llm = None
        svc._large_llm = None
        loaded = MagicMock(name="router")

        def _load():
            svc._llm = loaded

        svc.load_model.side_effect = _load

        assert _run_enrichment(svc) is loaded
        svc.load_model.assert_called_once()
        svc._get_large_llm.assert_not_called()

    def test_source_does_not_reference_the_lazy_loader(self):
        """Belt-and-braces against someone reinstating the lazy load — the
        MagicMock-based assertions above would silently pass if the call moved
        into a helper."""
        code = "\n".join(
            line for line in inspect.getsource(ModelService._enrich_mcp_server).splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "_get_large_llm" not in code
        assert "_large_llm" not in code
