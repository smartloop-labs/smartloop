import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from smartloop.db.workspace_indexer import WorkspaceIndexer


def _write_json(path: Path, rows: list) -> None:
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


class TestWorkspaceIndexerJsonToTable:
    def _make_indexer(self, work_dir: Path) -> WorkspaceIndexer:
        store = MagicMock()
        idx = WorkspaceIndexer.__new__(WorkspaceIndexer)
        idx._store = store
        idx._manifest = {}
        idx._logger = MagicMock()
        idx._base_dir = str(work_dir)
        idx._project_dir = str(work_dir)
        return idx

    def test_json_doc_indexed_as_table_not_raw_json(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            (work / "data.json").write_text(
                json.dumps({"rows": [{"name": "Drake Maye", "team": "NE", "td": 31}]}),
                encoding="utf-8",
            )
            idx = self._make_indexer(work)
            stats = idx._index(str(work))
            assert stats["indexed"] == 1

            added = idx._store.add_chunks.call_args[0][0]
            table_text = "\n".join(c["text"] for c in added)
            assert "| name | team | td |" in table_text
            assert "| Drake Maye | NE | 31 |" in table_text
            assert '"rows"' not in table_text
            idx._logger.info.assert_any_call(
                f"[workspace-index] data.json: converted JSON → table ({len(table_text)} chars)"
            )
