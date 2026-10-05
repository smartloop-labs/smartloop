//! Port of `tests/unit/test_workspace_db.py`.
//!
//! `#[serial]` throughout: timestamps come from the platform clock, which
//! other tests shift (`monkeypatch.setattr(time, "time", ...)`).

use serde_json::{json, Value};
use serial_test::serial;
use smartloop::platform::client::JsonObject;
use smartloop::platform::workspace_db::WorkspaceDB;

use crate::shared::platform_support::obj;

/// A WorkspaceDB backed by a temp file (the `db` fixture).
fn db() -> (tempfile::TempDir, WorkspaceDB) {
    let tmp = tempfile::tempdir().unwrap();
    let db = WorkspaceDB::new(tmp.path().join("workspace.db")).unwrap();
    (tmp, db)
}

fn user() -> JsonObject {
    obj(json!({
        "email": "chris@example.com",
        "name": "Chris",
        "is_premium": true,
        "is_paid": true,
        "number_of_tokens_processed": 42,
        "default_temperature": 0.7,
        "default_memory": true,
    }))
}

fn models() -> Vec<JsonObject> {
    vec![
        obj(json!({
            "model_id": "google/gemma-3-1b-it",
            "name": "gemma3-1b",
            "size": "1B",
            "description": "Gemma 3 1B",
            "access_level": "public",
            "is_accessible": true,
        })),
        obj(json!({
            "model_id": "google/gemma-3-4b-it",
            "name": "gemma3-4b",
            "size": "4B",
            "description": "Gemma 3 4B",
            "access_level": "paid",
            "is_accessible": false,
        })),
    ]
}

fn release() -> JsonObject {
    obj(json!({
        "version": "1.0.15",
        "name": "Release 1.0.15",
        "notes": "shiny",
        "channel": "stable",
        "artifacts": [
            {"platform": "darwin", "arch": "arm64", "kind": "zip",
             "url": "https://cdn/x.zip", "sha512": "abc", "size": 123},
        ],
        "min_version": "1.0.10",
    }))
}

fn with(mut base: JsonObject, changes: Value) -> JsonObject {
    for (k, v) in obj(changes) {
        base.insert(k, v);
    }
    base
}

mod test_workspace_db_creation {
    use super::*;

    #[test]
    #[serial]
    fn test_creates_db_file() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("sub").join("workspace.db");
        WorkspaceDB::new(&db_path).unwrap();
        assert!(db_path.exists());
    }

    #[test]
    #[serial]
    fn test_empty_on_creation() {
        let (_tmp, db) = db();
        assert!(db.get_user().unwrap().is_none());
        assert_eq!(db.list_models().unwrap(), Vec::<JsonObject>::new());
        assert!(db.user_fetched_at().unwrap().is_none());
        assert!(db.models_fetched_at().unwrap().is_none());
    }
}

mod test_user_persistence {
    use super::*;

    #[test]
    #[serial]
    fn test_save_and_get() {
        let (_tmp, db) = db();
        db.save_user(&user()).unwrap();
        let result = db.get_user().unwrap().unwrap();
        assert_eq!(result["email"], json!("chris@example.com"));
        assert_eq!(result["is_paid"], json!(true));
        assert_eq!(result["number_of_tokens_processed"], json!(42));
        assert_eq!(result["default_temperature"], json!(0.7));
    }

    #[test]
    #[serial]
    fn test_upsert_overwrites() {
        let (_tmp, db) = db();
        db.save_user(&user()).unwrap();
        let updated = with(user(), json!({"is_paid": false, "default_temperature": 0.4}));
        db.save_user(&updated).unwrap();
        let result = db.get_user().unwrap().unwrap();
        assert_eq!(result["is_paid"], json!(false));
        assert_eq!(result["default_temperature"], json!(0.4));
    }

    #[test]
    #[serial]
    fn test_fetched_at_set() {
        let (_tmp, db) = db();
        assert!(db.user_fetched_at().unwrap().is_none());
        db.save_user(&user()).unwrap();
        let ts = db.user_fetched_at().unwrap();
        assert!(ts.is_some());
        assert!(ts.unwrap() > 0.0);
    }

    #[test]
    #[serial]
    fn test_defaults_for_missing_fields() {
        let (_tmp, db) = db();
        db.save_user(&obj(json!({"email": "min@example.com"}))).unwrap();
        let result = db.get_user().unwrap().unwrap();
        assert_eq!(result["email"], json!("min@example.com"));
        assert_eq!(result["is_paid"], json!(false));
        assert_eq!(result["default_temperature"], json!(0.3));
        assert_eq!(result["default_memory"], json!(true));
    }
}

mod test_model_persistence {
    use super::*;

    #[test]
    #[serial]
    fn test_save_and_list() {
        let (_tmp, db) = db();
        db.save_models(&models()).unwrap();
        let result = db.list_models().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["model_id"], json!("google/gemma-3-1b-it"));
        assert_eq!(result[0]["is_accessible"], json!(true));
        assert_eq!(result[1]["is_accessible"], json!(false));
    }

    #[test]
    #[serial]
    fn test_save_replaces_all() {
        let (_tmp, db) = db();
        db.save_models(&models()).unwrap();
        db.save_models(&models()[..1]).unwrap();
        let result = db.list_models().unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    #[serial]
    fn test_fetched_at_set() {
        let (_tmp, db) = db();
        assert!(db.models_fetched_at().unwrap().is_none());
        db.save_models(&models()).unwrap();
        let ts = db.models_fetched_at().unwrap();
        assert!(ts.is_some());
        assert!(ts.unwrap() > 0.0);
    }

    #[test]
    #[serial]
    fn test_empty_save() {
        let (_tmp, db) = db();
        db.save_models(&models()).unwrap();
        db.save_models(&[]).unwrap();
        assert_eq!(db.list_models().unwrap(), Vec::<JsonObject>::new());
        assert!(db.models_fetched_at().unwrap().is_none());
    }
}

mod test_release_persistence {
    use super::*;

    #[test]
    #[serial]
    fn test_empty_on_creation() {
        let (_tmp, db) = db();
        assert!(db.get_release("stable").unwrap().is_none());
        assert!(db.get_release("beta").unwrap().is_none());
        assert!(db.release_fetched_at("stable").unwrap().is_none());
    }

    #[test]
    #[serial]
    fn test_save_and_get() {
        let (_tmp, db) = db();
        db.save_release("stable", Some(&release())).unwrap();
        assert_eq!(db.get_release("stable").unwrap(), Some(release()));
        assert!(db.release_fetched_at("stable").unwrap().unwrap() > 0.0);
    }

    #[test]
    #[serial]
    fn test_channels_are_independent() {
        let (_tmp, db) = db();
        db.save_release("stable", Some(&release())).unwrap();
        let beta = with(release(), json!({"channel": "beta", "version": "1.1.0-beta.1"}));
        db.save_release("beta", Some(&beta)).unwrap();
        assert_eq!(db.get_release("stable").unwrap().unwrap()["version"], json!("1.0.15"));
        assert_eq!(db.get_release("beta").unwrap().unwrap()["version"], json!("1.1.0-beta.1"));
    }

    #[test]
    #[serial]
    fn test_upsert_overwrites() {
        let (_tmp, db) = db();
        db.save_release("stable", Some(&release())).unwrap();
        db.save_release("stable", Some(&with(release(), json!({"version": "1.0.16"})))).unwrap();
        assert_eq!(db.get_release("stable").unwrap().unwrap()["version"], json!("1.0.16"));
    }

    #[test]
    #[serial]
    fn test_none_release_records_timestamp_but_returns_none() {
        let (_tmp, db) = db();
        db.save_release("stable", None).unwrap();
        assert!(db.get_release("stable").unwrap().is_none());
        assert!(db.release_fetched_at("stable").unwrap().unwrap() > 0.0); // a fetch DID happen
    }
}

mod test_workspace_persistence {
    use super::*;

    #[test]
    #[serial]
    fn test_empty_on_creation() {
        let (_tmp, db) = db();
        assert!(db.get_workspace().unwrap().is_none());
    }

    #[test]
    #[serial]
    fn test_save_and_get() {
        let (_tmp, db) = db();
        db.save_workspace("ws-uuid-123").unwrap();
        let result = db.get_workspace().unwrap().unwrap();
        assert_eq!(result["id"], json!("ws-uuid-123"));
        assert!(result["created_at"].as_f64().unwrap() > 0.0);
        assert_eq!(result["updated_at"], result["created_at"]);
    }

    #[test]
    #[serial]
    fn test_second_save_bumps_updated_at_only() {
        let (_tmp, db) = db();
        db.save_workspace("ws-uuid-123").unwrap();
        let first = db.get_workspace().unwrap().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        db.save_workspace("ws-uuid-123").unwrap();
        let second = db.get_workspace().unwrap().unwrap();
        assert_eq!(second["id"], first["id"]);
        assert_eq!(second["created_at"], first["created_at"]);
        assert!(second["updated_at"].as_f64().unwrap() > first["updated_at"].as_f64().unwrap());
    }
}
