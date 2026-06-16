"""
MCP Skill Generator — generates a single skill per MCP server
that covers all its tools, so the planner can route user requests
to the right MCP tools.

The tool listing is deterministic. An optional LLM call adds a plain-text
description of the server's purpose and what user requests match each tool.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

from smartloop.llms import BaseLLM
from smartloop.workspace.workspace import MCPServer, MCPToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class GeneratedSkill:
    name: str
    description: str
    content: str


class MCPSkillGenerator:
    """Generate a single skill from an MCP server's discovered tools."""

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm

    def generate_skills(self, server: MCPServer) -> List[GeneratedSkill]:
        """Generate a single skill covering all tools for the given MCP server."""
        if not server.tools:
            return []

        overview = self._generate_overview(server)
        content = self._build_skill_content(server, overview)

        skill = GeneratedSkill(
            name=server.name,
            description=f"When the user asks to interact with {server.name}, identify and invoke the matching tools.",
            content=content,
        )

        logger.info(
            "Generated skill for MCP server '%s' covering %d tool(s)",
            server.name, len(server.tools),
        )
        return [skill]

    def _generate_overview(self, server: MCPServer) -> Optional[str]:
        """Use the LLM to produce a plain-text overview of the server and its tools."""
        if not self.llm:
            return None

        tool_list = "\n".join(
            f"- {t.name}: {t.description or 'no description'}"
            for t in server.tools
        )

        prompt = f"""\
<system>
You are a technical documentation writer.
Write concise plain-text summaries. No markdown, no JSON.
</system>
<user>
The MCP server '{server.name}' provides the following tools:

{tool_list}

Write a short plain-text summary that describes:
1. The purpose of this server
2. For each tool, what kind of user request should use it

Be concise.
</user>"""

        try:
            response = self.llm.invoke(prompt)
            text = str(response).strip() if response else None
            if text:
                return text
        except Exception as e:
            logger.warning(
                "LLM overview generation failed for server '%s': %s",
                server.name, e,
            )
        return None

    @staticmethod
    def _build_skill_content(server: MCPServer, overview: Optional[str] = None) -> str:
        """Build markdown skill content from server metadata and optional LLM overview."""
        lines = [
            f"# {server.name}",
            "",
        ]

        if overview:
            lines.append(overview)
            lines.append("")
        elif server.description:
            lines.append(server.description)
            lines.append("")

        lines.append("## Available Tools")
        lines.append("")

        for tool in server.tools:
            lines.append(f"### `{tool.name}`")
            if tool.description:
                lines.append(tool.description)

            if tool.input_schema and tool.input_schema.get("properties"):
                params = tool.input_schema["properties"]
                required = set(tool.input_schema.get("required", []))
                lines.append("")
                lines.append("**Parameters:**")
                for pname, pinfo in params.items():
                    req = " *(required)*" if pname in required else ""
                    pdesc = pinfo.get("description", pinfo.get("type", ""))
                    lines.append(f"- `{pname}`{req}: {pdesc}")

            lines.append("")

        return "\n".join(lines)
