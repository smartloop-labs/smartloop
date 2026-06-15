"""FastMCP-based system tools server – bash, file-read, file-write, grep."""

import logging
import os
import re
import subprocess
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("smartloop")


def _extract_fenced_content(text: str) -> str:
    """Strip a single ```` ``` ```` fence (with optional language tag) that
    small models often wrap file content in, returning the inner text."""
    match = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


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

    content = _extract_fenced_content(content)

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


@mcp.tool(name="grep")
async def grep(pattern: str, glob: str = "*", timeout: int = 30) -> str:
    """Recursively search files in the working directory for a pattern.

    Args:
        pattern: Text or regex to search for.
        glob: Optional filename filter (e.g. "*.py"). Defaults to all files.
        timeout: Timeout in seconds.
    """
    if not pattern:
        return "Error: no search pattern provided"

    logger.info(f"[grep] Searching for '{pattern}' (glob={glob})")
    cmd = ["grep", "-rn", "--include", glob, pattern, "."]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Search timed out after {timeout}s"

    output = proc.stdout.strip()
    if not output:
        return f"No matches found for '{pattern}'"
    lines = output.split("\n")
    if len(lines) > 50:
        output = (
            "\n".join(lines[:50])
            + f"\n\n... ({len(lines)} total matches, showing first 50)"
        )
    return f"Search results for '{pattern}':\n\n{output}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
