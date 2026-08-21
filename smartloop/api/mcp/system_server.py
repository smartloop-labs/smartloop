"""FastMCP-based system tools server – bash, file-read, file-write, grep.

Tools raise (rather than return an "Error: ..." string) on failure so the
MCP result is flagged is_error=True.  The orchestrator then skips the step
instead of treating the error text as authoritative tool output, which
would otherwise suppress the workspace/document search fallback — e.g.
"find my pitch" with a hallucinated path or a grep that misses must fall
through to semantic search over the indexed files, not dead-end on
"file not found" / "no matches".
"""

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


def _clamp_cwd(cwd: Optional[str]) -> str:
    """Resolve a model-supplied ``cwd`` inside the project dir.

    The server's own working directory *is* ``project_dir`` — LocalMCPClient
    spawns this subprocess with ``cwd`` set to it — so it is the containment
    root.  A relative path resolves against it; an absolute or ``~`` path that
    lands outside it is dropped in favour of the root rather than honoured,
    the same way the API layer ignores the client-supplied ``cwd`` (it may
    point outside the sandbox container, or at a folder absent on a copied
    install).  A path inside the root that does not exist falls back to the
    root too, so bash still runs instead of failing to spawn.
    """
    root = os.path.realpath(os.getcwd())
    if not cwd:
        return root

    candidate = os.path.realpath(os.path.join(root, os.path.expanduser(cwd)))
    if candidate != root and not candidate.startswith(root + os.sep):
        logger.warning(
            "[bash] cwd %r resolves outside the project dir — using %s", cwd, root
        )
        return root
    if not os.path.isdir(candidate):
        logger.warning("[bash] cwd %r is not a directory — using %s", cwd, root)
        return root
    return candidate


@mcp.tool(name="bash")
async def bash(command: str, cwd: Optional[str] = None, timeout: int = 30) -> str:
    """Execute a bash command.

    Args:
        command: The bash command to execute.
        cwd: Working directory, relative to the project dir. Paths outside
            it are ignored and the project dir is used instead.
        timeout: Timeout in seconds.
    """
    work_dir = _clamp_cwd(cwd)
    logger.info(f"[bash] Executing: {command[:200]}, cwd={work_dir}")
    # A command that ran to completion is a real result even with a nonzero
    # exit code — its output (e.g. failing tests) belongs in the answer.
    # Only failures to produce a result at all (timeout, spawn error) raise.
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"command timed out after {timeout}s")

    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += proc.stderr
    return f"exit code {proc.returncode}\n{output.strip()}"


@mcp.tool(name="file-read")
async def file_read(path: str, query: Optional[str] = None) -> str:
    """Read a file from the working directory.

    Args:
        path: File path to read.
        query: Optional search query to filter matching lines.
    """
    if not path:
        raise ValueError("no file path provided")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {path}")

    logger.info(f"[file-read] Reading '{path}', query='{query or ''}'")

    try:
        content = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception as e:
        raise RuntimeError(f"error reading file: {e}")

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
        raise ValueError("no file path provided")
    if content is None:
        raise ValueError("no content provided")

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
        raise RuntimeError(f"error writing file: {e}")


@mcp.tool(name="grep")
async def grep(pattern: str, glob: str = "*", timeout: int = 30) -> str:
    """Recursively search files in the working directory for a pattern.

    Args:
        pattern: Text or regex to search for.
        glob: Optional filename filter (e.g. "*.py"). Defaults to all files.
        timeout: Timeout in seconds.
    """
    if not pattern:
        raise ValueError("no search pattern provided")

    logger.info(f"[grep] Searching for '{pattern}' (glob={glob})")
    cmd = ["grep", "-rn", "--include", glob, pattern, "."]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"search timed out after {timeout}s")

    # grep exits 0 on match, 1 on no match, >=2 on error (stdout is empty
    # either way — without the returncode check a bad regex would be
    # misreported as a clean miss).
    if proc.returncode > 1:
        raise RuntimeError(f"grep failed: {proc.stderr.strip()}")

    output = proc.stdout.strip()
    if not output:
        raise RuntimeError(f"no matches found for '{pattern}'")
    lines = output.split("\n")
    if len(lines) > 50:
        output = (
            "\n".join(lines[:50])
            + f"\n\n... ({len(lines)} total matches, showing first 50)"
        )
    return f"Search results for '{pattern}':\n\n{output}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
