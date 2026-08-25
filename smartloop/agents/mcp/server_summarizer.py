"""
MCP Server Summarizer — generates a concise summary of an MCP server
and its tools to help the LLM pick the right tool at selection time.
"""
import json
import logging
import re
from typing import Dict, List, Optional

from smartloop.llms import BaseLLM
from smartloop.workspace.workspace import MCPServer

logger = logging.getLogger(__name__)


class MCPServerSummarizer:
    """Summarize an MCP server and its tools into a short paragraph."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_metadata(self, server: MCPServer) -> Dict[str, object]:
        """Generate a short description and topical tags for an MCP server.

        Used to enrich hand-added connections (e.g. ``uvx`` packages) that
        connect without advertising a description. Returns
        ``{"description": str, "tags": list[str]}`` — an empty dict on failure
        so the caller can degrade gracefully. Mirrors the JSON-parsing approach
        used for skill metadata generation.
        """
        if not server.tools:
            return {}

        tools_desc = "\n".join(
            f"- {t.name}: {t.description or 'no description'}"
            for t in server.tools
        )
        prompt = (
            "<system>\n"
            "You describe MCP tool servers. Reply with ONLY a JSON object — "
            "no markdown, no explanation.\n"
            "</system>\n"
            "<user>\n"
            "Given this MCP server and its tools, output a JSON object with:\n"
            '- "description": one clear sentence describing what this server does '
            "and what its tools let the assistant do, based only on the tools "
            "listed below.\n"
            '- "tags": an array of 1 to 3 short, lowercase, single-word topical '
            "tags derived from this server's tools.\n\n"
            f"Name: {server.name}\n"
            f"Package: {server.pkg or 'none'}\n"
            f"Existing description: {server.description or 'none'}\n"
            f"Tools:\n{tools_desc}\n"
            "</user>"
        )
        try:
            raw = str(self.llm.invoke(prompt)).strip()
            logger.info("MCP metadata LLM response for '%s': %s", server.name, raw[:200])

            # Strip markdown fences
            if raw.startswith("```"):
                raw = "\n".join(ln for ln in raw.split("\n") if not ln.strip().startswith("```")).strip()

            # Parse the FIRST JSON object from the first "{", ignoring any
            # trailing text or extra objects the model appends (raw_decode
            # stops at the end of the first value — plain json.loads would raise
            # "Extra data" on those).
            start = raw.find("{")
            if start < 0:
                return {}
            raw = raw[start:]
            # Fix invalid JSON escapes that LLMs produce (e.g. \_ \*)
            raw = re.sub(r'\\([^"\\/bfnrtu])', r'\1', raw)
            meta, _ = json.JSONDecoder().raw_decode(raw)

            description = meta.get("description")
            tags = meta.get("tags")
            result: Dict[str, object] = {}
            if isinstance(description, str) and description.strip():
                # Collapse newlines/control chars/runs of whitespace into single
                # spaces so the stored description is always clean one-line text.
                result["description"] = " ".join(description.split())
            if isinstance(tags, list):
                clean = [str(t).strip().lower() for t in tags if str(t).strip()]
                if clean:
                    result["tags"] = clean[:3]
            return result
        except Exception as e:
            logger.warning(
                "Failed to generate metadata for MCP server '%s': %s",
                server.name, e,
            )
            return {}

    def invoke(self, server: MCPServer) -> Optional[str]:
        """Generate a concise summary for the given MCP server.

        Returns:
            A short summary string, or None if summarization fails.
        """
        if not server.tools:
            return None

        tools_desc = "\n".join(
            f"- {t.name}: {t.description or 'no description'}"
            for t in server.tools
        )
        prompt = (
            "<system>You summarize MCP tool servers in one concise paragraph. "
            "Focus on what the server does and what its tools can accomplish. "
            "Keep it under 100 words.</system>\n"
            f"<user>Summarize this MCP server:\n"
            f"Name: {server.name}\n"
            f"Description: {server.description or 'none'}\n"
            f"Tools:\n{tools_desc}</user>"
        )
        try:
            prompt_tokens = self.llm.get_num_tokens(prompt)
            logger.info(
                "Server summary prompt (%d tokens) for '%s'",
                prompt_tokens, server.name,
            )

            response = self.llm.invoke(prompt)
            summary = str(response).strip() if response else None
            if summary:
                response_tokens = self.llm.get_num_tokens(summary)
                logger.info(
                    "Generated summary for MCP server '%s' (%d tokens): %s",
                    server.name, response_tokens, summary[:100],
                )
            return summary
        except Exception as e:
            logger.warning(
                "Failed to generate summary for MCP server '%s': %s",
                server.name, e,
            )
            return None
