"""Render JSON (``.json``/``.jsonl`` files, or JSON embedded in ``.txt``) into
GitHub-flavored Markdown.

MarkItDown's plain-text converter accepts ``application/json`` but returns the
raw bytes verbatim — it does not structure JSON.  This parser parses the JSON
and emits nested bullet lists, turning arrays of uniform objects into Markdown
tables so the content indexes as readable structure rather than one blob.
Malformed JSON falls back to the raw file text.
"""
import json
import os

from .base_parser import BaseParser
from smartloop.paths import expand_home


def looks_like_json(file_path: str) -> bool:
    """Cheap sniff: does the file's first non-space char start a JSON value?

    Only used to route ``.txt`` uploads; a real ``.json`` extension routes here
    directly.  Reads a small prefix so a large plain-text file is not fully
    loaded just to reject it.
    """
    try:
        with open(expand_home(file_path), "r", encoding="utf-8") as fh:
            head = fh.read(64)
    except (OSError, UnicodeDecodeError):
        return False
    stripped = head.lstrip()
    return bool(stripped) and stripped[0] in "{["


class JsonParser(BaseParser):

    def parse(self) -> str:
        filepath = expand_home(self.file_path)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as fh:
            raw = fh.read()

        data, is_jsonl = self._load(raw)
        if data is None:
            # Not valid JSON after all — index the raw text rather than fail.
            if self.logger:
                self.logger.info(
                    "JsonParser: %s is not valid JSON, indexing as raw text",
                    filepath,
                )
            return raw.strip()

        if is_jsonl:
            data = {"records": data}
        return self._to_markdown(data).strip()

    def _load(self, raw: str):
        """Return ``(data, is_jsonl)``; ``data`` is ``None`` when unparseable."""
        try:
            return json.loads(raw), False
        except json.JSONDecodeError:
            pass
        # Try JSON Lines: one JSON value per non-empty line.
        records = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return None, False
        return (records, True) if records else (None, False)

    # -- markdown rendering -------------------------------------------------

    def _to_markdown(self, value, indent: int = 0) -> str:
        if isinstance(value, dict):
            return self._render_dict(value, indent)
        if isinstance(value, list):
            return self._render_list(value, indent)
        return f"{self._pad(indent)}{self._scalar(value)}"

    def _render_dict(self, obj: dict, indent: int) -> str:
        lines = []
        pad = self._pad(indent)
        for key, val in obj.items():
            if isinstance(val, list) and self._is_table(val):
                # A table must sit at the left margin with a blank line before
                # and after it, or Markdown renders the pipes as literal text
                # instead of a table.  Keep the key as a label above it.
                lines.append(f"{pad}- **{key}**:")
                lines.append("")
                lines.append(self._render_table(val))
                lines.append("")
            elif isinstance(val, (dict, list)) and val:
                lines.append(f"{pad}- **{key}**:")
                lines.append(self._to_markdown(val, indent + 1))
            else:
                lines.append(f"{pad}- **{key}**: {self._scalar(val)}")
        return "\n".join(lines)

    def _render_list(self, arr: list, indent: int) -> str:
        # An array of uniform objects (no nested containers) renders as a table.
        if self._is_table(arr):
            return self._render_table(arr)

        lines = []
        pad = self._pad(indent)
        for item in arr:
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}-")
                lines.append(self._to_markdown(item, indent + 1))
            else:
                lines.append(f"{pad}- {self._scalar(item)}")
        return "\n".join(lines)

    @staticmethod
    def _is_table(arr: list) -> bool:
        if len(arr) < 1 or not all(isinstance(i, dict) for i in arr):
            return False
        # Every cell must be a scalar so rows stay on one line.
        return all(
            not isinstance(v, (dict, list))
            for row in arr for v in row.values()
        )

    def _render_table(self, arr: list) -> str:
        # Rendered at the left margin (no indent): an indented pipe-table is
        # not recognized as a table by GitHub-flavored Markdown.
        columns = []
        for row in arr:
            for key in row:
                if key not in columns:
                    columns.append(key)

        lines = [
            f"| {' | '.join(columns)} |",
            f"| {' | '.join(['---'] * len(columns))} |",
        ]
        for row in arr:
            cells = [self._cell(row.get(col, "")) for col in columns]
            lines.append(f"| {' | '.join(cells)} |")
        return "\n".join(lines)

    @staticmethod
    def _pad(indent: int) -> str:
        return "  " * indent

    @staticmethod
    def _scalar(value) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _cell(self, value) -> str:
        # Escape pipes so a value never breaks the table layout.
        return self._scalar(value).replace("|", "\\|").replace("\n", " ")
