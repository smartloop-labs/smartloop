//! Port of `tests/unit/test_workspace_indexer.py`.

use std::path::Path;
use std::sync::Arc;

use smartloop::db::workspace_indexer::WorkspaceIndexer;

use crate::shared::db::{capture_logs, RecordingStore};

mod test_workspace_indexer_json_to_table {
    use super::*;

    /// An indexer over a recording store with an empty manifest (the Python
    /// test builds one with `__new__` and a `MagicMock` store).
    fn make_indexer(work_dir: &Path) -> (WorkspaceIndexer, Arc<RecordingStore>) {
        let store = RecordingStore::new();
        let idx = WorkspaceIndexer::with_store(work_dir.to_str().unwrap(), store.clone());
        idx.set_manifest(Default::default());
        (idx, store)
    }

    #[test]
    fn test_json_doc_indexed_as_table_not_raw_json() {
        let td = tempfile::tempdir().unwrap();
        let work = td.path();
        std::fs::write(
            work.join("data.json"),
            serde_json::to_string(&serde_json::json!({
                "rows": [{"name": "Drake Maye", "team": "NE", "td": 31}]
            }))
            .unwrap(),
        )
        .unwrap();
        let (idx, store) = make_indexer(work);
        let (stats, logs) = capture_logs(|| idx.index_inner(work.to_str().unwrap()));
        assert_eq!(stats.indexed, 1);

        let added = store.last_added().expect("add_chunks was called");
        let table_text = added
            .iter()
            .map(|c| c["text"].as_str().unwrap().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(table_text.contains("| name | team | td |"));
        assert!(table_text.contains("| Drake Maye | NE | 31 |"));
        assert!(!table_text.contains("\"rows\""));
        let expected = format!(
            "[workspace-index] data.json: converted JSON → table ({} chars)",
            table_text.chars().count()
        );
        assert!(logs.has_info(&expected), "missing log line {expected:?}: {:?}", logs.records());
    }
}
