"""The workspace index must not feed on its own output.

``IndexerTask`` points the walk at ``project_dir`` — the framework's own state
directory — and the indexer writes its manifest there at the end of every run.
Indexing that manifest made each run dirty the input to the next: one file
re-embedded every scan, forever, for nothing.
"""

import json
import os
import tempfile
import threading
from unittest.mock import MagicMock

from smartloop.db.workspace_indexer import WorkspaceIndexer, _MANIFEST_FILENAME


def _indexer(project_dir):
    idx = WorkspaceIndexer.__new__(WorkspaceIndexer)
    idx._project_dir = project_dir
    idx._logger = MagicMock()
    idx._store = MagicMock()
    idx._manifest_path = os.path.join(project_dir, _MANIFEST_FILENAME)
    idx._manifest = {}
    idx._lock = threading.Lock()
    return idx


class TestStateFilesAreNotContent:
    def test_the_indexer_does_not_index_its_own_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, _MANIFEST_FILENAME), "w") as fh:
                json.dump({"a": "b"}, fh)
            with open(os.path.join(td, "notes.md"), "w") as fh:
                fh.write("real content")

            idx = _indexer(td)
            idx.index(td)
            assert set(idx._manifest) == {"notes.md"}

    def test_a_second_run_over_unchanged_content_indexes_nothing(self):
        # The loop this exists to prevent: run one wrote the manifest, so run
        # two must still see nothing to do.
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "notes.md"), "w") as fh:
                fh.write("real content")

            idx = _indexer(td)
            first = idx.index(td)
            second = idx.index(td)
            assert first["indexed"] == 1
            assert second["indexed"] == 0
            assert second["skipped"] == 1

    def test_other_hidden_files_are_skipped_too(self):
        # The MCP crawl writes its own manifest into the same directory, and
        # that one churns on every crawl.
        with tempfile.TemporaryDirectory() as td:
            for name in (".mcp_documents_manifest.json", ".env", ".DS_Store"):
                with open(os.path.join(td, name), "w") as fh:
                    fh.write("{}")
            idx = _indexer(td)
            idx.index(td)
            assert idx._manifest == {}

    def test_the_vector_store_directory_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "db"))
            with open(os.path.join(td, "db", "embedding_model.txt"), "w") as fh:
                fh.write("bge-m3")
            idx = _indexer(td)
            idx.index(td)
            assert idx._manifest == {}

    def test_a_db_directory_deeper_in_the_tree_is_still_indexed(self):
        # Only the scan root's `db` is ours; a project with its own db/ of
        # source files keeps it.
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src", "db"))
            with open(os.path.join(td, "src", "db", "schema.sql"), "w") as fh:
                fh.write("create table t (id int);")
            idx = _indexer(td)
            idx.index(td)
            assert set(idx._manifest) == {os.path.join("src", "db", "schema.sql")}
