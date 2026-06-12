import json
import logging
from typing import Any, Dict, List, Optional

from smartloop.llms import BaseLLM

logger = logging.getLogger(__name__)


class ResultFormatter:
    """Formats MCP tool results into compact, readable markdown."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def format_tool_result(
        self,
        tool_name: str,
        content: Optional[List[dict]],
        is_error: bool,
    ) -> str:
        if not content:
            return f"Tool '{tool_name}' completed with no output."

        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get('type', 'text')
                if block_type == 'text':
                    text = block.get('text', '')
                    try:
                        data = json.loads(text)
                        data = self.compact_json(data)
                        parts.append(self.flatten_to_text(data))
                    except (json.JSONDecodeError, TypeError):
                        parts.append(text)
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))

        result_text = "\n".join(parts)

        if is_error:
            return f"**Error** from `{tool_name}`:\n{result_text}"

        max_tokens = int(self.llm.get_max_tokens() * 0.6)
        if self.llm.get_num_tokens(result_text) > max_tokens:
            cutoff = max_tokens * 4
            result_text = result_text[:cutoff] + "\n\n... (truncated)"

        return result_text

    @staticmethod
    def is_noisy_value(key: str, value) -> bool:
        if value is None or isinstance(value, bool):
            return True
        key_lower = key.lower()
        noisy_keys = (
            'id', 'uid', 'uuid', 'guid', '_id', 'node_id',
            'url', 'href', 'uri', 'link', 'api_url',
            'avatar', 'icon', 'logo', 'thumbnail', 'image_url', 'gravatar',
            'sha', 'hash', 'etag', 'checksum',
            'created_at', 'updated_at', 'deleted_at', 'timestamp',
            'created', 'modified', 'closed_at', 'merged_at', 'pushed_at',
            'permissions', 'node_id', 'events_url', 'hooks_url',
        )
        if key_lower in noisy_keys:
            return True
        if key_lower.endswith(('_url', '_uri', '_href', '_id', '_at', '_sha')):
            return True
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(('http://', 'https://', 'ssh://', 'git@')):
                return True
            if len(stripped) >= 32 and all(c in '0123456789abcdefABCDEF' for c in stripped):
                return True
        return False

    @classmethod
    def compact_json(cls, data):
        if isinstance(data, list):
            return [cls.compact_json(item) for item in data]
        if isinstance(data, dict):
            compacted = {}
            for key, value in data.items():
                if cls.is_noisy_value(key, value):
                    continue
                if isinstance(value, dict):
                    nested = cls.compact_json(value)
                    if nested:
                        compacted[key] = nested
                elif isinstance(value, list):
                    compacted[key] = cls.compact_json(value)
                else:
                    compacted[key] = value
            # Unwrap single-child wrapper dicts (e.g. GraphQL envelopes
            # like {"data": {"issues": {"nodes": [...]}}}).
            if len(compacted) == 1:
                only_value = next(iter(compacted.values()))
                if isinstance(only_value, (dict, list)):
                    return cls.compact_json(only_value)
            return compacted
        return data

    @staticmethod
    def _humanize_key(key: str) -> str:
        """Convert camelCase / snake_case keys to readable words.

        regularMarketPrice  →  regular market price
        fifty_two_week_high →  fifty two week high
        """
        import re
        # camelCase → insert space before each uppercase run
        key = re.sub(r'([a-z])([A-Z])', r'\1 \2', key)
        key = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', key)
        return key.replace("_", " ").lower()

    @staticmethod
    def flatten_to_text(data, _prefix: str = "") -> str:
        """Convert a compacted dict/list to plain key: value lines.

        Produces output like:
            symbol: AAPL
            regular market price: 290.64
            volume: 25520114
        so the LLM never sees raw JSON syntax and can answer naturally.
        """
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                label = ResultFormatter._humanize_key(key)
                if _prefix:
                    label = _prefix + label
                if isinstance(value, (dict, list)):
                    nested = ResultFormatter.flatten_to_text(value, _prefix=label + " / ")
                    if nested:
                        lines.append(nested)
                else:
                    lines.append(f"{label}: {value}")
            return "\n".join(lines)
        if isinstance(data, list):
            chunks = []
            for item in data:
                if isinstance(item, (dict, list)):
                    chunks.append(ResultFormatter.flatten_to_text(item, _prefix=_prefix))
                else:
                    chunks.append(f"{_prefix}{item}" if _prefix else str(item))
            return "\n".join(chunks)
        return f"{_prefix}{data}" if _prefix else str(data)

    @staticmethod
    def json_to_markdown(data) -> str:
        if isinstance(data, list):
            rows = [item for item in data if isinstance(item, dict)]
            if not rows:
                return "\n".join(str(item) for item in data)

            columns: list[str] = []
            seen: set[str] = set()
            for row in rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        columns.append(key)

            def _is_renderable(val) -> bool:
                if not isinstance(val, (dict, list)):
                    return True
                if isinstance(val, dict) and (
                    "name" in val or "title" in val or len(val) == 1
                ):
                    return True
                return False

            scalar_cols = [
                col for col in columns
                if any(_is_renderable(row.get(col)) for row in rows)
            ]
            if not scalar_cols:
                scalar_cols = columns[:5]

            header = "| " + " | ".join(scalar_cols) + " |"
            sep = "| " + " | ".join("---" for _ in scalar_cols) + " |"
            lines = [header, sep]
            for row in rows:
                cells = []
                for col in scalar_cols:
                    val = row.get(col, "")
                    if val is None:
                        val = ""
                    if isinstance(val, dict):
                        if "name" in val:
                            val = val["name"]
                        elif "title" in val:
                            val = val["title"]
                        elif len(val) == 1:
                            val = next(iter(val.values()))
                        else:
                            val = ", ".join(f"{k}: {v}" for k, v in val.items())
                    elif isinstance(val, list):
                        val = ", ".join(str(v) for v in val[:5])
                    cell = str(val).replace("|", "\\|").replace("\n", " ")
                    if len(cell) > 80:
                        cell = cell[:77] + "..."
                    cells.append(cell)
                lines.append("| " + " | ".join(cells) + " |")
            return "\n".join(lines)

        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    nested = ResultFormatter.json_to_markdown(value)
                    lines.append(f"**{key}**:\n{nested}")
                else:
                    val_str = str(value) if value is not None else ""
                    lines.append(f"- **{key}**: {val_str}")
            return "\n".join(lines)

        return str(data)
