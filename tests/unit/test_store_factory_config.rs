//! Port of `tests/unit/test_store_factory_config.py`.
//!
//! Tests for the vector store factory.
//!
//! SQLite (dense embeddings, TF-IDF fallback) is the only backend; the
//! factory always builds an `IndexStore` regardless of the `store_type`
//! argument, which is retained only for call-site compatibility.

use smartloop::db::sqlite_store::IndexStore;
use smartloop::db::store::VectorStore;
use smartloop::db::store_factory::{resolve_default_store_type, StoreFactory, StoreType};

fn project_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

fn is_index_store(store: &dyn VectorStore) -> bool {
    store
        .as_any()
        .is_some_and(|any| any.downcast_ref::<IndexStore>().is_some())
}

mod test_factory_resolution {
    use super::*;

    #[test]
    fn test_default_store_type_is_sqlite() {
        assert_eq!(resolve_default_store_type(), StoreType::SQLite);
    }

    #[test]
    fn test_factory_builds_sqlite_by_default() {
        let dir = project_dir();
        let store =
            StoreFactory::create_store(dir.path().to_str().unwrap(), "documents", None, None).unwrap();
        assert!(is_index_store(store.as_ref()));
    }

    #[test]
    fn test_store_type_argument_is_ignored() {
        // The legacy store_type arg is accepted but always yields an IndexStore.
        let dir = project_dir();
        let store = StoreFactory::create_store(
            dir.path().to_str().unwrap(),
            "documents",
            None,
            Some("anything"),
        )
        .unwrap();
        assert!(is_index_store(store.as_ref()));
    }
}
