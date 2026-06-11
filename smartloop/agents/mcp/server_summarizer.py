"""
MCP Server Summarizer — generates a concise summary of an MCP server
and its tools to help the LLM pick the right tool at selection time.
"""
import logging
from typing import List, Optional

from smartloop.llms import BaseLLM
from smartloop.workspace.workspace import MCPServer

logger = logging.getLogger(__name__)


class MCPServerSummarizer:
    """Summarize an MCP server and its tools into a short paragraph."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

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
