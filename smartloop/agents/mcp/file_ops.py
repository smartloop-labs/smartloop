import json
import logging
import mimetypes
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from smartloop.llms import BaseLLM
from smartloop.parsers import ParserFactory

from .server_selector import GREP_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL

logger = logging.getLogger(__name__)


class FileOperations:
    """Local file I/O operations: read, write, grep, bash, and binary parsing."""

    BINARY_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv"}

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def read_file(self, args: Dict[str, str], work_dir: str, user_message: str) -> str:
        file_path = args.get("path")
        if not file_path:
            return "Error: no file path provided"

        full_path = os.path.join(work_dir, file_path) if not os.path.isabs(file_path) else file_path
        logger.info(f"[file-read] Reading '{full_path}', query='{args.get('query', '')}'")

        if not os.path.isfile(full_path):
            return f"Error: file not found: {file_path}"

        ext = os.path.splitext(full_path)[1].lower()

        if ext in self.BINARY_EXTENSIONS:
            content = self.parse_binary_file(full_path)
            if content is None:
                return f"Error: could not parse {ext} file: {file_path}"
            logger.info(f"[file-read] Parsed {ext} file ({len(content)} chars)")
        else:
            try:
                content = open(full_path, "r", encoding="utf-8", errors="replace").read()
            except Exception as e:
                return f"Error reading file: {e}"

        query = args.get("query") or ""
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
                return f"File: {file_path}\nMatching lines for '{query}':\n\n" + "\n".join(matching)

        if len(content) > 4000:
            return f"File: {file_path} ({len(content)} chars)\n\n{content[:4000]}\n\n... (truncated)"
        return f"File: {file_path}\n\n{content}"

    def parse_binary_file(self, file_path: str) -> Optional[str]:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            return None

        try:
            factory = ParserFactory(llm=self.llm)
            parser = factory.get_parser(mime_type, file_path, logger)
            return parser.parse()
        except ValueError:
            logger.warning(f"[file-read] No parser for mime type: {mime_type}")
            return None
        except Exception as e:
            logger.error(f"[file-read] Parser failed for {file_path}: {e}")
            return None

    @staticmethod
    def extract_fenced_content(text: str) -> str:
        match = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    @staticmethod
    def write_file(args: Dict[str, str], work_dir: str) -> str:
        file_path = args.get("path")
        file_content = args.get("content")
        if not file_path:
            return "Error: no file path provided"
        if file_content is None:
            return "Error: no content provided"

        file_content = FileOperations.extract_fenced_content(file_content)

        if os.path.isabs(file_path):
            try:
                rel = os.path.relpath(file_path, work_dir)
                if rel.startswith(".."):
                    file_path = os.path.basename(file_path)
                    logger.warning(f"[file-write] Redirecting out-of-cwd path to '{file_path}'")
                    full_path = os.path.join(work_dir, file_path)
                else:
                    full_path = file_path
            except ValueError:
                full_path = os.path.join(work_dir, os.path.basename(file_path))
        else:
            full_path = os.path.join(work_dir, file_path)

        logger.info(f"[file-write] Writing to '{full_path}', content_len={len(file_content)}")
        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            return (
                f"Successfully wrote to {file_path} ({len(file_content)} chars). "
                "IMPORTANT: Reference this file path in your response so the user "
                "can locate and download it from the chat log."
            )
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    def grep(args: Dict[str, str], work_dir: str, timeout: int = 30) -> str:
        pattern = args.get("pattern")
        if not pattern:
            return "Error: no search pattern provided"

        glob_filter = args.get("glob", "*")
        logger.info(f"[grep] Searching for '{pattern}' (glob={glob_filter}) in {work_dir}")
        cmd = ["grep", "-rn", "--include", glob_filter, pattern, "."]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=work_dir,
            )
            output = proc.stdout.strip()
            if not output:
                return f"No matches found for '{pattern}'"
            lines = output.split("\n")
            if len(lines) > 50:
                output = "\n".join(lines[:50]) + f"\n\n... ({len(lines)} total matches, showing first 50)"
            return f"Search results for '{pattern}':\n\n{output}"
        except subprocess.TimeoutExpired:
            return f"Search timed out after {timeout}s"

    @staticmethod
    def execute_bash(blocks: list[str], timeout: int = 30, cwd: Optional[str] = None) -> str:
        logger.info(f"[bash] Executing {len(blocks)} block(s), cwd={cwd}")
        results = []
        for i, block in enumerate(blocks):
            logger.info(f"[bash] Block {i+1}: {block.strip()[:200]}")
            try:
                proc = subprocess.run(
                    ["bash", "-c", block.strip()],
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
                results.append(
                    f"[Command {i+1}] exit code {proc.returncode}\n{output.strip()}"
                )
            except subprocess.TimeoutExpired:
                results.append(f"[Command {i+1}] timed out after {timeout}s")
            except Exception as e:
                results.append(f"[Command {i+1}] error: {e}")
        return "\n\n".join(results)

    def extract_builtin_args(
        self,
        tool_name: str,
        user_message: str,
        previous_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, str]]:
        schemas = {
            GREP_TOOL: 'Extract: "pattern" (search text/regex), "glob" (optional file filter e.g. "*.py").',
            FILE_READ_TOOL: 'Extract: "path" (file path to read), "query" (what the user is looking for in the file).',
            FILE_WRITE_TOOL: (
                'Extract: "path" (file path to write — look for a "FOUND FILES:" section in '
                'the previous tool results and use the most relevant path listed there; '
                'fall back to inferring from any file-read result or the user request), '
                '"content" (the COMPLETE updated text to write — you MUST synthesise '
                'this from the previous tool results to fulfil the user\'s request; '
                'do NOT leave this null or empty).'
            ),
        }

        context_section = ""
        if previous_results:
            parts = []
            for r in previous_results:
                t_name = r.get("tool_name", "unknown")
                result_text = r.get("result", "")
                if len(result_text) > 3000:
                    result_text = result_text[:3000] + "\n... (truncated)"
                parts.append(f"[{t_name} result]\n{result_text}")
            context_section = (
                "Previous tool results (use these to determine arguments):\n"
                + "\n\n".join(parts)
                + "\n\n"
            )

        prompt = (
            f"User request: {user_message}\n\n"
            f"{context_section}"
            f"Tool: {tool_name}\n"
            f"{schemas.get(tool_name, '')}\n\n"
            'Reply with ONLY a JSON object containing the extracted fields. '
            'Use null for fields you cannot determine.'
        )
        try:
            response = str(self.llm.invoke(prompt)).strip()
            if response.startswith("```"):
                lines = response.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                response = "\n".join(lines).strip()
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Failed to extract args for {tool_name}: {e}")
            return None
