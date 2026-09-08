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
from smartloop.workspace.workspace import PROJECT_STATE_FILE


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
        # Other passes write their own manifests into the same directory.
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

    def test_the_project_state_file_is_not_indexed(self):
        # ``project.json`` is the same mistake as the manifest without the
        # leading dot: the project's own record, rewritten every turn, so
        # indexing it re-embedded a megabyte of bookkeeping on every scan.
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, PROJECT_STATE_FILE), "w") as fh:
                json.dump({"project": {"id": "p1", "documents": []}}, fh)
            with open(os.path.join(td, "notes.md"), "w") as fh:
                fh.write("real content")

            idx = _indexer(td)
            idx.index(td)
            assert set(idx._manifest) == {"notes.md"}

    def test_rewriting_the_state_file_leaves_the_index_alone(self):
        # The loop as the user felt it: every turn rewrites project.json, so
        # every scan used to delete and re-embed all of it.
        with tempfile.TemporaryDirectory() as td:
            state = os.path.join(td, PROJECT_STATE_FILE)
            with open(state, "w") as fh:
                json.dump({"project": {"sessions": []}}, fh)
            with open(os.path.join(td, "notes.md"), "w") as fh:
                fh.write("real content")

            idx = _indexer(td)
            idx.index(td)
            with open(state, "w") as fh:
                json.dump({"project": {"sessions": ["a", "b", "c"]}}, fh)
            second = idx.index(td)

            assert second["indexed"] == 0
            assert second["removed"] == 0

    def test_a_state_file_deeper_in_the_tree_is_still_indexed(self):
        # Only the scan root's copy is ours — a project that happens to keep a
        # project.json of its own under src/ keeps it.
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src"))
            nested = os.path.join(td, "src", PROJECT_STATE_FILE)
            with open(nested, "w") as fh:
                json.dump({"name": "someone else's file"}, fh)
            idx = _indexer(td)
            idx.index(td)
            assert set(idx._manifest) == {os.path.join("src", PROJECT_STATE_FILE)}

    def test_an_already_indexed_state_file_is_removed_from_the_store(self):
        # Existing installs have it in the store and the manifest; the first
        # pass after the fix has to clean up rather than orphan 567 chunks.
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, PROJECT_STATE_FILE), "w") as fh:
                json.dump({"project": {}}, fh)
            idx = _indexer(td)
            idx._manifest = {PROJECT_STATE_FILE: "whatever-it-hashed-to"}

            stats = idx.index(td)

            assert stats["removed"] == 1
            assert idx._manifest == {}
            idx._store.delete_document.assert_called_once()
