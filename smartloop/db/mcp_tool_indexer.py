"""
MCPToolIndexer — vectorizes the tools of a registered MCP server into a
dedicated per-project vector collection ("mcp_tools") so an MCP step can
retrieve the most relevant tools for an instruction by semantic similarity.

One vector is stored per tool.  All tools of a server share the same
``document_id`` (the server id) so removing a server deletes its tools in a
single :meth:`VectorStore.delete_document` call.  Each tool row carries the
fields required to resolve and execute it later:

* ``description``  — embedded as the searchable text (with the name).
* ``qualified_name`` — ``server.tool_name`` (the display/lookup name).
* ``input_schema`` — the tool's JSON schema (stored as a JSON string).
* ``tool_name`` / ``server_id`` / ``server_name`` — the tool call target.
"""
import json
import logging
from typing import List, Optional

from smartloop.db.store import ChunkResult
from smartloop.db.store_factory import StoreFactory

logger = logging.getLogger(__name__)

MCP_TOOLS_COLLECTION = "mcp_tools"


class MCPToolIndexer:
    """Indexes MCP tools into a per-project ``mcp_tools`` vector collection."""

    def __init__(
        self,
        project_dir: str,
        logger: Optional[logging.Logger] = None,
        base_dir: Optional[str] = None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._store = StoreFactory.create_store(
            project_dir=project_dir,
            collection_name=MCP_TOOLS_COLLECTION,
            logger=self._logger,
            base_dir=base_dir,
        )

    def index_server(self, server) -> int:
        """Upsert one vector per tool for *server*.

        *server* is a workspace ``MCPServer`` with ``id``, ``name`` and a
        ``tools`` list of ``MCPToolDefinition`` (``name``, ``description``,
        ``input_schema``).  Returns the number of tools indexed.
        """
        from smartloop.db.embeddings import indexing_burst

        with indexing_burst():
            return self._index_server(server)

    def _index_server(self, server) -> int:
        # Clear any stale tools for this server before re-adding (re-register
        # with a changed tool set must not leave orphans behind).
        try:
            self._store.delete_document(server.id)
        except Exception:
            pass

        chunks: List[dict] = []
        for i, tool in enumerate(getattr(server, "tools", None) or []):
            qualified = f"{server.name}.{tool.name}"
            description = tool.description or tool.name
            chunks.append({
                # Embedded text: qualified name + description drives retrieval.
                "text": f"{qualified}: {description}",
                "document_id": server.id,
                "chunk_index": i,
                "document_name": qualified,
                # Execution/resolution metadata (str/int/float are persisted).
                "tool_name": tool.name,
                "server_id": server.id,
                "server_name": server.name,
                "qualified_name": qualified,
                "input_schema": json.dumps(tool.input_schema or {}),
            })

        if not chunks:
            return 0

        try:
            self._store.add_chunks(chunks)
            self._logger.info(
                f"[mcp-index] Indexed {len(chunks)} tool(s) for server "
                f"{server.name} ({server.id})"
            )
        except Exception as e:
            self._logger.warning(f"[mcp-index] Failed to index {server.name}: {e}")
            return 0
        return len(chunks)

    def count(self) -> int:
        """Return the total number of indexed tool vectors for this project."""
        try:
            return self._store.count()
        except Exception as e:
            self._logger.warning(f"[mcp-index] count failed: {e}")
            return 0

    def remove_server(self, server_id: str) -> None:
        """Delete all indexed tools for a server."""
        try:
            self._store.delete_document(server_id)
            self._logger.info(f"[mcp-index] Removed tools for server {server_id}")
        except Exception as e:
            self._logger.debug(f"[mcp-index] remove_server({server_id}) failed: {e}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        max_distance: float = 0.8,
    ) -> List[ChunkResult]:
        """Return the top candidate tools for *query* by semantic similarity.

        Each result's ``metadata`` carries ``qualified_name``, ``tool_name``,
        ``server_id``, ``server_name`` and ``input_schema`` (JSON string).
        """
        try:
            return self._store.query_with_metadata(
                query, n_results=n_results, max_distance=max_distance
            )
        except Exception as e:
            self._logger.warning(f"[mcp-index] Search failed: {e}")
            return []
