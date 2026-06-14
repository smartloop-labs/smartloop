"""FastMCP-based system tools server – bash, file-read, file-write."""

import logging
import os
import subprocess
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("smartloop")


@mcp.tool(name="bash")
async def bash(command: str, cwd: Optional[str] = None, timeout: int = 30) -> str:
    """Execute a bash command.

    Args:
        command: The bash command to execute.
        cwd: Working directory for the command.
        timeout: Timeout in seconds.
    """
    logger.info(f"[bash] Executing: {command[:200]}, cwd={cwd}")
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = ""
        if proc.stdout:
            output += proc.stdout
        if proc.stderr:
            output += proc.stderr
        return f"exit code {proc.returncode}\n{output.strip()}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool(name="file-read")
async def file_read(path: str, query: Optional[str] = None) -> str:
    """Read a file from the working directory.

    Args:
        path: File path to read.
        query: Optional search query to filter matching lines.
    """
    if not path:
        return "Error: no file path provided"

    if not os.path.isfile(path):
        return f"Error: file not found: {path}"

    logger.info(f"[file-read] Reading '{path}', query='{query or ''}'")

    try:
        content = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception as e:
        return f"Error reading file: {e}"

    if query:
        lines = content.split("\n")
        matching = []
        query_lower = query.lower()
        for i, line in enumerate(lines, 1):
            if query_lower in line.lower():
                start = max(0, i - 3)
                end = min(len(lines), i + 2)
                for j in range(start, end):
                    entry = f"{j+1}: {lines[j]}"
                    if entry not in matching:
                        matching.append(entry)
                if len(matching) > 100:
                    break
        if matching:
            return (
                f"File: {path}\nMatching lines for '{query}':\n\n"
                + "\n".join(matching)
            )

    if len(content) > 4000:
        return (
            f"File: {path} ({len(content)} chars)\n\n{content[:4000]}"
            "\n\n... (truncated)"
        )
    return f"File: {path}\n\n{content}"


@mcp.tool(name="file-write")
async def file_write(path: str, content: str) -> str:
    """Write content to a file in the working directory.

    Args:
        path: File path to write.
        content: Content to write.
    """
    if not path:
        return "Error: no file path provided"
    if content is None:
        return "Error: no content provided"

    logger.info(f"[file-write] Writing to '{path}', content_len={len(content)}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return (
            f"Successfully wrote to {path} ({len(content)} chars). "
            "IMPORTANT: Reference this file path in your response so the user "
            "can locate and download it from the chat log."
        )
    except Exception as e:
        return f"Error writing file: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
