"""Document map-reduce: one DAG node per retrieved document.

EVERY retrieved document gets its own node, so the answer covers them all and
nothing is dropped for volume.  Processing each document independently is also
what makes per-document relevance possible: a document the query doesn't ask
about is filtered out by its own ``NONE`` extraction instead of leaking into a
co-batched document's result.

The reduce step merges the surviving extracts and narrows the citation list to
exactly the documents that contributed.
"""
import logging
from typing import Any, Callable, Dict, List, Optional

from smartloop.agents.nodes import (
    AgentNode, LLMDecisionNode, NodeResult, TransformNode,
)
from smartloop.agents.mcp.explorer import SearchContext
from smartloop.agents.orchestration.context import _OrchestratorContext
from smartloop.agents.orchestration.model_router import ModelRouter
from smartloop.agents.orchestration.replies import (
    is_none_extract, is_placeholder_output,
)
from smartloop.agents.orchestration.step import Step, StepNodes
from smartloop.agents.planner import ModelTier
from smartloop.config.app_settings import AppSettings
from smartloop.types import ModelCapability

logger = logging.getLogger(__name__)


class DocumentFanoutStep(Step):
    """Builds and runs the per-document extract → reduce sub-DAG."""

    name = "documents"

    # When at least this many documents match, each document is processed in
    # its own DAG node and reduced into a single result.  The number processed
    # concurrently is bounded by the LLM worker pool; overflow queues.
    THRESHOLD = 1

    def __init__(self, models: ModelRouter):
        self._models = models

    def applies_to(
        self,
        ctx: _OrchestratorContext,
        search_ctx: Optional[SearchContext],
    ) -> bool:
        """True when the retrieved documents warrant the fan-out."""
        return bool(search_ctx) and len(search_ctx.documents) >= self.THRESHOLD

    def _batch_documents(self, documents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Map each document to its own batch (one document per fan-out node).

        EVERY retrieved document gets its own node so the answer covers them
        all — nothing is dropped regardless of count.  Processing each document
        independently also means a document the query doesn't ask about (e.g.
        another person's invoice) is filtered out by its own ``NONE`` extraction
        instead of leaking into a co-batched document's result.

        The number processed concurrently is bounded by the LLM worker pool
        (and the DAG thread pool); any overflow simply waits in the pool's
        queue and is processed as workers free up.
        """
        return [[doc] for doc in documents]

    def build(
        self,
        ctx: _OrchestratorContext,
        search_ctx: SearchContext,
        parent: AgentNode,
    ) -> StepNodes:
        """Build the document map-reduce sub-DAG.

        The reduce node becomes the parent for everything downstream, so later
        steps see the merged extract rather than racing the individual batches.
        """
        skill_instruction = ctx.skill_context

        batches = self._batch_documents(search_ctx.documents)
        logger.info(
            f"[doc-fanout] {len(search_ctx.documents)} document(s) → "
            f"{len(batches)} batch node(s) (serialized in-thread)"
        )

        nodes_and_handlers: List[tuple] = []
        batch_nodes: List[LLMDecisionNode] = []
        for i, batch in enumerate(batches):
            node = LLMDecisionNode(depends_on=[parent])
            handler = self._make_batch_handler(
                ctx, node.node_id, batch, skill_instruction,
            )
            nodes_and_handlers.append((node, handler))
            batch_nodes.append(node)

        batch_node_ids = [n.node_id for n in batch_nodes]
        reduce_node = TransformNode(depends_on=batch_nodes)
        reduce_handler = self.make_reduce_handler(ctx, search_ctx, batch_node_ids)
        nodes_and_handlers.append((reduce_node, reduce_handler))

        return StepNodes(pairs=nodes_and_handlers, parent=reduce_node)

    @staticmethod
    def _json_to_table(text: str) -> Optional[str]:
        """See :func:`smartloop.utils.doc_splitter.json_to_table`."""
        from smartloop.utils import json_to_table
        return json_to_table(text)

    @staticmethod
    def _is_markdown_table(text: str) -> bool:
        """True when the text contains a complete markdown table.

        See :func:`smartloop.utils.doc_splitter.is_markdown_table`.
        """
        from smartloop.utils import is_markdown_table
        return is_markdown_table(text)

    def _make_batch_handler(
        self,
        ctx: _OrchestratorContext,
        node_id: str,
        batch: List[Dict[str, Any]],
        skill_instruction: Optional[str],
    ) -> Callable:
        """Create a handler that extracts query-relevant content from one batch.

        Inference uses the shared in-process model, serialized upstream by
        the DAG executor's LLM lock.
        """
        # Reserve room for prompt scaffolding + output.  The chars/token
        # estimate is a conservative floor for data-dense content (stats,
        # JSON tables) — prose runs closer to 4 chars/token but dense data
        # can drop to ~2, and an optimistic estimate overflows the window
        # and kills extraction entirely.
        _CHARS_PER_TOKEN = 2.0
        _OUTPUT_RESERVE_TOKENS = 1024

        def handler(deps: Dict) -> NodeResult:
            doc_names = [d.get("document_name", "") for d in batch]
            _named = [n for n in doc_names if n]
            self._models.emit(
                "read", "processing",
                f"Reading document: {', '.join(_named)}" if _named else "Reading document",
            )
            combined = "\n\n".join(
                f"=== Document: {d.get('document_name', '')} ===\n{d.get('content', '')}"
                for d in batch
            )

            # Structured data documents are used deterministically — no LLM
            # context window, so large files can't overflow the extractor, and
            # exact values are preserved.  Two shapes are handled:
            #   * raw JSON (an array of objects, e.g. a {"rows": [...]} dump)
            #     → converted to a table on the fly;
            #   * content already normalized to a markdown table at index time
            #     → used directly.
            # Both skip loading the DP model entirely.
            table_parts = []
            for d in batch:
                doc_name = d.get("document_name", "")
                content = d.get("content", "")
                t = self._json_to_table(content)
                if t is None and self._is_markdown_table(content):
                    t = content
                if t is not None:
                    table_parts.append(f"{doc_name}\n{t}" if doc_name else t)
            if table_parts:
                table = "\n\n---\n\n".join(table_parts)
                logger.info(
                    f"[doc-batch] {node_id}: json→table conversion → "
                    f"{len(table)} chars"
                )
                return NodeResult(node_id=node_id, success=True, data={
                    "doc_names": doc_names,
                    "content": table,
                })

            # Size the extraction input to the EXTRACTION model's actual
            # context window — not the router's.  The router's max_tokens can
            # exceed the DP model's window (e.g. 8192 vs 6080), so sizing from
            # it overflows the extractor and generation silently fails.
            dp_llm = None
            max_chars = 16000
            try:
                dp_llm = self._models.llm_for_tier(ModelTier.LARGE, capability=ModelCapability.DP)
                logger.info(
                    "[doc-batch] %s → %s",
                    node_id, self._models.model_label(dp_llm),
                )
                ctx_tokens = int(dp_llm.get_max_tokens() or 0)
                if ctx_tokens > 0:
                    usable = max(512, ctx_tokens - _OUTPUT_RESERVE_TOKENS)
                    max_chars = int(usable * _CHARS_PER_TOKEN)
                _s = AppSettings()
                if _s.doc_extract_max_chars > 0:
                    max_chars = min(max_chars, _s.doc_extract_max_chars)
            except Exception as e:
                logger.warning(f"[doc-batch] {node_id} context sizing failed: {e}")

            prompt = (
                "<system>\n"
                "You extract the information relevant to the user's request from "
                "the document(s) below. Preserve concrete facts, numbers, names, "
                "dates, and identifiers exactly as written. Be faithful to the "
                "source — do not invent or infer beyond it. Return a concise "
                "factual extract; if nothing is relevant, reply with NONE.\n"
                "</system>\n"
                "<user>\n"
                f"User request: {ctx.user_message}\n\n"
                + (f"Guidance:\n{skill_instruction}\n\n" if skill_instruction else "")
                + f"Documents:\n{combined[:max_chars]}\n"
                "</user>"
            )

            extract = ""
            llm_failed = False
            try:
                extract = str(dp_llm.invoke(prompt)).strip()
            except Exception as e:
                llm_failed = True
                logger.warning(f"[doc-batch] {node_id} extraction failed: {e}")

            if is_none_extract(extract):
                # The LLM judged this document irrelevant to the request (it was
                # told to reply NONE).  Respect that and drop the batch so an
                # entity-specific query ("items Adam Hart purchased") doesn't
                # pull in other people's documents.
                logger.info(
                    f"[doc-batch] {node_id}: not relevant → dropped "
                    f"({', '.join(n for n in doc_names if n)})"
                )
                extract = ""
            elif not extract or llm_failed or is_placeholder_output(extract):
                # No usable extract: the model errored, returned empty, or
                # produced only a placeholder (e.g. llama.cpp's empty-response
                # fallback "No response generated" — which also happens when the
                # extraction prompt overflows the model's context window and
                # generation silently fails).  Fall back to a raw excerpt so a
                # degenerate model never replaces the document with placeholder
                # text.
                logger.info(
                    f"[doc-batch] {node_id}: extract unusable → falling back "
                    f"to raw excerpt ({len(extract)} chars)"
                )
                extract = combined[:2000]

            logger.info(
                f"[doc-batch] {node_id}: {len(batch)} doc(s) "
                f"({', '.join(n for n in doc_names if n)}) → {len(extract)} chars"
            )
            return NodeResult(node_id=node_id, success=True, data={
                "doc_names": doc_names,
                "content": extract,
            })

        return handler

    def make_reduce_handler(
        self,
        ctx: _OrchestratorContext,
        search_ctx: SearchContext,
        batch_node_ids: List[str],
    ) -> Callable:
        """Create the reduce handler that merges batch extracts into one document."""
        def handler(deps: Dict) -> NodeResult:
            # Merge in batch order (most relevant first) for determinism.
            # Each extract is prefixed with its source document name(s) so the
            # downstream LLM can attribute every fact to the right document
            # instead of anchoring the whole answer on a single source.
            parts: List[str] = []
            surviving_names: List[str] = []  # docs that actually contributed
            # name -> the extract fed to the model for that document.  Carried
            # on the citation so the router can check the finished answer
            # actually drew on it before crediting the source.
            extract_by_name: Dict[str, str] = {}
            for node_id in batch_node_ids:
                dep_result = deps.get(node_id)
                if dep_result and dep_result.success and isinstance(dep_result.data, dict):
                    content = dep_result.data.get("content", "")
                    if content:
                        names = [n for n in dep_result.data.get("doc_names", []) if n]
                        surviving_names.extend(names)
                        for n in names:
                            extract_by_name[n] = content
                        header = ", ".join(names) if names else "Unknown source"
                        parts.append(f"=== Source: {header} ===\n{content}")

            merged = "\n\n---\n\n".join(parts)
            # Replace the raw concatenated context with the reduced,
            # query-focused document so every downstream step sees it.
            #
            # An empty merge means the per-document judge dropped every
            # retrieved document as irrelevant (each replied NONE).  That
            # verdict has to clear the context, not leave it alone: keeping
            # the raw concatenation resurrected exactly the documents the
            # fan-out just rejected, and the answer was then framed as
            # DOCUMENT CONTEXT built from them — so the model reported it had
            # no context for the question rather than answering it directly.
            if merged:
                search_ctx.vector_context = merged
            else:
                search_ctx.vector_context = ""
                search_ctx.citations = None
                search_ctx.cited_documents = None
                logger.info(
                    "[doc-reduce] every retrieved document judged irrelevant "
                    "— cleared document context"
                )

            # Bundle citations from ONLY the surviving (relevant) documents.
            # Build the list directly from the surviving doc names (deduped, in
            # relevance order) rather than filtering the full citation list —
            # so we never accidentally keep citations for documents the fan-out
            # dropped (which happened when a name didn't match and the filter
            # fell back to the full set).  Reuse page info from the original
            # citations when a name matches; synthesize a minimal entry
            # otherwise.  Record the surviving set so the final citation list
            # stays restricted even if a later EXPLORE step re-collects.
            if surviving_names:
                ordered = list(dict.fromkeys(surviving_names))
                search_ctx.cited_documents = ordered
                by_name: Dict[str, Dict[str, Any]] = {}
                for c in (search_ctx.citations or []):
                    name = c.get("document_name")
                    if name and name not in by_name:
                        by_name[name] = c
                search_ctx.citations = []
                for name in ordered:
                    citation = dict(by_name.get(name, {"document_name": name}))
                    # Only fill in an extract when the citation has no source
                    # text yet.  The retrieval citation already carries the
                    # document's own words, which is the better evidence: this
                    # extract is the DP model's answer *to the user's query*,
                    # so it echoes the query's wording and would match the
                    # final answer no matter which document it came from.
                    # Private, stripped before the citation reaches a client.
                    if not citation.get("_extract"):
                        extract = extract_by_name.get(name)
                        if extract:
                            citation["_extract"] = extract
                    search_ctx.citations.append(citation)

            logger.info(
                f"[doc-reduce] merged {len(parts)} batch extract(s) → {len(merged)} chars; "
                f"cited {len(search_ctx.citations or [])} surviving doc(s)"
            )
            return NodeResult(node_id="doc_reduce", success=True, data={
                "content": merged,
                "search_ctx": search_ctx,
                "citations": search_ctx.citations,
            })

        return handler

    # ── Skill pipeline node builders ──