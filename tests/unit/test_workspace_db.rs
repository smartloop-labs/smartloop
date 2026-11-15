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
    fn test_remote_model_round_trips() {
        let (_tmp, db) = db();
        let remote = obj(json!({
            "model_id": null,
            "name": "kimi-k2",
            "access_level": "free_login",
            "is_accessible": true,
            "type": "remote",
            "url": "https://api.smartloop.ai",
            "capabilities": ["general", "coding"],
            "context_length": 131072,
        }));
        let mut rows = models();
        rows.push(remote);
        db.save_models(&rows).unwrap();

        let result = db.list_models().unwrap();
        let cached = result.iter().find(|m| m["name"] == json!("kimi-k2")).unwrap();
        assert_eq!(cached["model_id"], Value::Null);
        assert_eq!(cached["type"], json!("remote"));
        assert_eq!(cached["url"], json!("https://api.smartloop.ai"));
        assert_eq!(cached["capabilities"], json!(["general", "coding"]));
        assert_eq!(cached["context_length"], json!(131072));
        let local = result.iter().find(|m| m["name"] == json!("gemma3-1b")).unwrap();
        // Part of the cache key, so an untyped row is stored as local.
        assert_eq!(local["type"], json!("local"));
        assert!(local.get("capabilities").is_none());
    }

    #[test]
    #[serial]
    fn test_spec_fields_round_trip() {
        let (_tmp, db) = db();
        let mut rows = models();
        rows[0].extend(obj(json!({
            "param_size_b": 1.5,
            "param_size_estimated": true,
            "active_param_size_b": 0.5,
            "attention": "hybrid",
            "reasoning_budget_tokens": 1024,
            "model_version": 3,
            "thinking_default": true,
            "vision_chat_handler": "MTMDChatHandler",
        })));
        db.save_models(&rows).unwrap();

        let result = db.list_models().unwrap();
        let declared = result.iter().find(|m| m["name"] == json!("gemma3-1b")).unwrap();
        assert_eq!(declared["param_size_b"], json!(1.5));
        assert_eq!(declared["param_size_estimated"], json!(true));
        assert_eq!(declared["active_param_size_b"], json!(0.5));
        assert_eq!(declared["attention"], json!("hybrid"));
        assert_eq!(declared["reasoning_budget_tokens"], json!(1024));
        assert_eq!(declared["model_version"], json!(3));
        assert_eq!(declared["thinking_default"], json!(true));
        assert_eq!(declared["vision_chat_handler"], json!("MTMDChatHandler"));
        // Undeclared stays null rather than reading as false or 0.
        let bare = result.iter().find(|m| m["name"] == json!("gemma3-4b")).unwrap();
        for key in [
            "param_size_b",
            "param_size_estimated",
            "active_param_size_b",
            "attention",
            "reasoning_budget_tokens",
            "model_version",
            "thinking_default",
            "vision_chat_handler",
        ] {
            assert_eq!(bare[key], Value::Null, "{key}");
        }
    }

    #[test]
    #[serial]
    fn test_a_catalog_cached_before_the_reasoning_budget_is_discarded() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("workspace.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE model_catalog (
                     name TEXT PRIMARY KEY, model_id TEXT, size TEXT,
                     description TEXT, access_level TEXT,
                     is_accessible INTEGER NOT NULL DEFAULT 0, sha256 TEXT,
                     type TEXT, url TEXT, capabilities TEXT, context_length INTEGER,
                     param_size_b REAL, model_version INTEGER, thinking_default INTEGER,
                     vision_chat_handler TEXT, param_size_estimated INTEGER,
                     active_param_size_b REAL, attention TEXT,
                     fetched_at REAL NOT NULL);
                 INSERT INTO model_catalog (model_id, name, fetched_at)
                     VALUES ('google/gemma-3-1b-it', 'gemma3-1b', 1.0);",
            )
            .unwrap();
        }

        let db = WorkspaceDB::new(&path).unwrap();
        assert!(db.list_models().unwrap().is_empty());
        db.save_models(&models()).unwrap();
        assert_eq!(db.list_models().unwrap().len(), 2);
    }

    #[test]
    #[serial]
    fn test_a_catalog_cached_before_remote_models_is_discarded() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("workspace.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE model_catalog (
                     model_id TEXT PRIMARY KEY, name TEXT NOT NULL, size TEXT,
                     description TEXT, access_level TEXT,
                     is_accessible INTEGER NOT NULL DEFAULT 0, sha256 TEXT,
                     fetched_at REAL NOT NULL);
                 INSERT INTO model_catalog (model_id, name, fetched_at)
                     VALUES ('google/gemma-3-1b-it', 'gemma3-1b', 1.0);",
            )
            .unwrap();
        }

        let db = WorkspaceDB::new(&path).unwrap();
        assert!(db.list_models().unwrap().is_empty());
        db.save_models(&models()).unwrap();
        assert_eq!(db.list_models().unwrap().len(), 2);
    }

    #[test]
    #[serial]
    fn test_a_name_listed_remote_and_local_keeps_both_rows() {
        let (_tmp, db) = db();
        let rows: Vec<JsonObject> = [
            json!({"name": "qwen3.8-27b", "model_id": null, "type": "remote"}),
            json!({"name": "qwen3.8-27b", "model_id": "Qwen/Qwen3.8-27B", "type": "local"}),
            json!({"name": "gemma3-1b", "model_id": "google/gemma-3-1b-it"}),
        ]
        .into_iter()
        .map(|v| v.as_object().unwrap().clone())
        .collect();
        db.save_models(&rows).unwrap();
        let result = db.list_models().unwrap();
        let kinds: Vec<_> = result.iter().map(|m| (m["name"].clone(), m["type"].clone())).collect();
        assert_eq!(
            kinds,
            vec![
                (json!("qwen3.8-27b"), json!("remote")),
                (json!("qwen3.8-27b"), json!("local")),
                // A row without a type is a local model's.
                (json!("gemma3-1b"), json!("local")),
            ]
        );
    }

    #[test]
    #[serial]
    fn test_an_older_clients_catalog_is_neither_read_nor_touched() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("workspace.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE model_catalog (
                     name TEXT PRIMARY KEY, model_id TEXT, size TEXT,
                     description TEXT, access_level TEXT,
                     is_accessible INTEGER NOT NULL DEFAULT 0, sha256 TEXT,
                     type TEXT, url TEXT, capabilities TEXT, context_length INTEGER,
                     param_size_b REAL, model_version INTEGER, thinking_default INTEGER,
                     vision_chat_handler TEXT, param_size_estimated INTEGER,
                     active_param_size_b REAL, attention TEXT,
                     reasoning_budget_tokens INTEGER,
                     fetched_at REAL NOT NULL);
                 INSERT INTO model_catalog (model_id, name, fetched_at)
                     VALUES ('google/gemma-3-1b-it', 'gemma3-1b', 1.0);",
            )
            .unwrap();
        }

        // An older client sharing the DB fetched one row per name into
        // `model_catalog`; this client reads its own table, and leaves that
        // one for the older client.
        let db = WorkspaceDB::new(&path).unwrap();
        assert!(db.list_models().unwrap().is_empty());
        db.save_models(&models()).unwrap();
        drop(db);
        let conn = rusqlite::Connection::open(&path).unwrap();
        let old: i64 = conn.query_row("SELECT COUNT(*) FROM model_catalog", [], |r| r.get(0)).unwrap();
        assert_eq!(old, 1);
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

mod test_connection_settings {
    use super::*;

    #[test]
    #[serial]
    fn test_the_file_is_in_wal_mode() {
        let (tmp, _db) = db();
        let conn = rusqlite::Connection::open(tmp.path().join("workspace.db")).unwrap();
        let mode: String = conn.query_row("PRAGMA journal_mode", [], |r| r.get(0)).unwrap();
        assert_eq!(mode, "wal");
    }

    #[test]
    #[serial]
    fn test_foreign_keys_are_enforced() {
        let (_tmp, db) = db();
        let err = db
            .write(|tx| {
                tx.execute_batch(
                    "CREATE TABLE parent (id TEXT PRIMARY KEY);
                     CREATE TABLE child (parent_id TEXT REFERENCES parent(id));
                     INSERT INTO child VALUES ('missing');",
                )?;
                Ok(())
            })
            .unwrap_err();
        assert!(err.to_string().contains("FOREIGN KEY"), "{err}");
    }
}

mod test_writes {
    use super::*;

    #[test]
    #[serial]
    fn test_a_failed_write_leaves_nothing_behind() {
        let (_tmp, db) = db();
        let result: anyhow::Result<()> = db.write(|tx| {
            tx.execute("INSERT INTO workspace (id, created_at, updated_at) VALUES ('ws', 1, 1)", [])?;
            anyhow::bail!("stop")
        });
        assert!(result.is_err());
        assert!(db.get_workspace().unwrap().is_none());
    }

    #[test]
    #[serial]
    fn test_two_processes_write_at_once() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("workspace.db");
        // Two handles on one file stand in for the supervisor and a child.
        let handles: Vec<_> = (0..2).map(|_| std::sync::Arc::new(WorkspaceDB::new(&path).unwrap())).collect();
        let writers: Vec<_> = (0..8)
            .map(|i| {
                let db = handles[i % 2].clone();
                std::thread::spawn(move || {
                    for n in 0..25 {
                        let release = obj(json!({"version": format!("{i}.{n}")}));
                        db.save_release(&format!("channel-{i}"), Some(&release)).unwrap();
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
        for i in 0..8 {
            let release = handles[0].get_release(&format!("channel-{i}")).unwrap().unwrap();
            assert_eq!(release["version"], json!(format!("{i}.24")));
        }
    }
}

mod test_reads {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::{Duration, Instant};

    #[test]
    #[serial]
    fn test_a_read_does_not_wait_on_an_open_write() {
        let (_tmp, db) = db();
        let db = Arc::new(db);
        db.save_workspace("before").unwrap();
        let (started, wait_for_start) = mpsc::channel();
        let writer = {
            let db = db.clone();
            std::thread::spawn(move || {
                db.write(|tx| {
                    tx.execute("UPDATE workspace SET id = 'during'", [])?;
                    started.send(()).unwrap();
                    std::thread::sleep(Duration::from_millis(500));
                    Ok(())
                })
                .unwrap();
            })
        };
        wait_for_start.recv().unwrap();
        let begun = Instant::now();
        let ws = db.get_workspace().unwrap().unwrap();
        assert!(begun.elapsed() < Duration::from_millis(250), "read waited {:?}", begun.elapsed());
        assert_eq!(ws["id"], json!("before"));
        writer.join().unwrap();
        assert_eq!(db.get_workspace().unwrap().unwrap()["id"], json!("during"));
    }

    #[test]
    #[serial]
    fn test_a_read_sees_another_handles_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("workspace.db");
        let reader = WorkspaceDB::new(&path).unwrap();
        assert!(reader.get_workspace().unwrap().is_none());
        WorkspaceDB::new(&path).unwrap().save_workspace("ws-1").unwrap();
        assert_eq!(reader.get_workspace().unwrap().unwrap()["id"], json!("ws-1"));
    }
}

mod test_shared_handle {
    use super::*;
    use smartloop::platform::{get_platform_service, workspace_db_at};
    use smartloop::workspace::workspace::Workspace;

    use crate::shared::workspace_support::TempHome;

    #[test]
    #[serial]
    fn test_one_handle_per_path() {
        let tmp = tempfile::tempdir().unwrap();
        let first = tmp.path().join("a").join("workspace.db");
        let second = tmp.path().join("b").join("workspace.db");
        let a = workspace_db_at(&first).unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &workspace_db_at(&first).unwrap()));
        let b = workspace_db_at(&second).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&a, &b));
        assert_eq!(b.db_path(), second.as_path());
    }

    #[test]
    #[serial]
    fn test_the_platform_service_binds_the_stored_workspace() {
        let home = TempHome::new();
        let ws_dir = home.join("ws-1");
        Workspace::new("ws-1").with_workspace_dir(ws_dir.clone()).save().unwrap();

        let service = get_platform_service();
        let db = service.db().expect("bound to the workspace DB");
        let path = std::path::Path::new(&ws_dir).join("workspace.db");
        assert_eq!(db.db_path(), path.as_path());
        assert!(std::sync::Arc::ptr_eq(&db, &workspace_db_at(&path).unwrap()));
        assert_eq!(db.get_workspace().unwrap().unwrap()["id"], json!("ws-1"));
    }
}

mod test_migrations {
    use super::*;
    use rusqlite::Connection;
    use smartloop::platform::workspace_db::apply_migrations;

    const STEPS: &[&str] = &["CREATE TABLE one (id TEXT)", "CREATE TABLE two (id TEXT)"];

    fn version(conn: &Connection) -> i64 {
        conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap()
    }

    fn tables(conn: &Connection) -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('one', 'two', 'three') ORDER BY name")
            .unwrap();
        stmt.query_map([], |r| r.get(0)).unwrap().collect::<Result<_, _>>().unwrap()
    }

    #[test]
    fn test_a_new_workspace_db_has_every_step() {
        let (tmp, _db) = db();
        let conn = Connection::open(tmp.path().join("workspace.db")).unwrap();
        assert_eq!(version(&conn), 1);
        let durable: Vec<String> = conn
            .prepare("SELECT name FROM pragma_table_list WHERE strict = 1 AND schema = 'main' ORDER BY name")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(durable, ["accounts", "projects", "workspace_members", "workspaces"]);
    }

    #[test]
    fn test_reopening_a_migrated_file_keeps_its_version() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("workspace.db");
        WorkspaceDB::new(&path).unwrap().projects().add("ws", "p1").unwrap();
        let db = WorkspaceDB::new(&path).unwrap();
        assert_eq!(db.projects().ids("ws").unwrap(), ["p1"]);
        assert_eq!(version(&Connection::open(&path).unwrap()), 1);
    }

    #[test]
    fn test_steps_apply_once_in_order() {
        let mut conn = Connection::open_in_memory().unwrap();
        apply_migrations(&mut conn, STEPS).unwrap();
        assert_eq!(version(&conn), 2);
        assert_eq!(tables(&conn), ["one", "two"]);

        // The steps aren't IF NOT EXISTS, so running one twice would fail.
        apply_migrations(&mut conn, STEPS).unwrap();
        let more = [STEPS[0], STEPS[1], "CREATE TABLE three (id TEXT)"];
        apply_migrations(&mut conn, &more).unwrap();
        assert_eq!(version(&conn), 3);
        assert_eq!(tables(&conn), ["one", "three", "two"]);
    }

    #[test]
    fn test_a_failing_step_applies_nothing() {
        let mut conn = Connection::open_in_memory().unwrap();
        assert!(apply_migrations(&mut conn, &[STEPS[0], "NOT SQL"]).is_err());
        assert_eq!(version(&conn), 0);
        assert!(tables(&conn).is_empty());
    }

    #[test]
    fn test_a_file_from_a_newer_client_is_left_alone() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "user_version", 5).unwrap();
        apply_migrations(&mut conn, STEPS).unwrap();
        assert_eq!(version(&conn), 5);
        assert!(tables(&conn).is_empty());
    }
}

mod test_projects {
    use super::*;
    use smartloop::platform::service::set_clock;

    /// Restores the real clock when dropped.
    struct RealClock;

    impl Drop for RealClock {
        fn drop(&mut self) {
            set_clock(None);
        }
    }

    /// Stop the platform clock at *seconds*.
    fn clock_at(seconds: f64) -> RealClock {
        set_clock(Some(std::sync::Arc::new(move || seconds)));
        RealClock
    }

    fn created_at(db: &WorkspaceDB, project_id: &str) -> i64 {
        db.read(|conn| {
            Ok(conn.query_row("SELECT created_at FROM projects WHERE id = ?", [project_id], |r| r.get(0))?)
        })
        .unwrap()
    }

    fn count(db: &WorkspaceDB, table: &str) -> i64 {
        db.read(|conn| Ok(conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))?))
            .unwrap()
    }

    fn sorted(mut ids: Vec<String>) -> Vec<String> {
        ids.sort();
        ids
    }

    #[test]
    #[serial]
    fn test_add_records_the_project_and_its_workspace() {
        let (_tmp, db) = db();
        db.projects().add("ws", "p1").unwrap();
        assert_eq!(db.projects().ids("ws").unwrap(), ["p1"]);
        assert_eq!(count(&db, "workspaces"), 1);
        let migrated: i64 = db
            .read(|conn| Ok(conn.query_row("SELECT mcp_migrated FROM projects WHERE id = 'p1'", [], |r| r.get(0))?))
            .unwrap();
        assert_eq!(migrated, 0);
    }

    #[test]
    #[serial]
    fn test_no_account_is_assigned() {
        let (_tmp, db) = db();
        db.projects().add("ws", "p1").unwrap();
        assert_eq!(count(&db, "accounts"), 0);
        assert_eq!(count(&db, "workspace_members"), 0);
    }

    #[test]
    #[serial]
    fn test_adding_twice_keeps_the_first_row() {
        let (_tmp, db) = db();
        let _clock = clock_at(1000.0);
        db.projects().add("ws", "p1").unwrap();
        let _clock = clock_at(2000.0);
        db.projects().add("ws", "p1").unwrap();
        assert_eq!(db.projects().ids("ws").unwrap(), ["p1"]);
        assert_eq!(created_at(&db, "p1"), 1000);
    }

    #[test]
    #[serial]
    fn test_ids_are_per_workspace() {
        let (_tmp, db) = db();
        db.projects().add("ws-a", "p1").unwrap();
        db.projects().add("ws-b", "p2").unwrap();
        assert_eq!(db.projects().ids("ws-a").unwrap(), ["p1"]);
        assert_eq!(db.projects().ids("ws-b").unwrap(), ["p2"]);
    }

    #[test]
    #[serial]
    fn test_remove_deletes_only_that_project() {
        let (_tmp, db) = db();
        db.projects().add("ws", "p1").unwrap();
        db.projects().add("ws", "p2").unwrap();
        assert!(db.projects().remove("p1").unwrap());
        assert!(!db.projects().remove("p1").unwrap());
        assert_eq!(db.projects().ids("ws").unwrap(), ["p2"]);
        assert_eq!(count(&db, "workspaces"), 1);
    }

    #[test]
    #[serial]
    fn test_remove_cascades_to_rows_keyed_on_the_project() {
        let (_tmp, db) = db();
        db.projects().add("ws", "p1").unwrap();
        db.write(|tx| {
            tx.execute_batch(
                "CREATE TABLE child (project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE);
                 INSERT INTO child VALUES ('p1');",
            )?;
            Ok(())
        })
        .unwrap();
        db.projects().remove("p1").unwrap();
        assert_eq!(count(&db, "child"), 0);
    }

    #[test]
    #[serial]
    fn test_a_wrongly_typed_value_is_rejected() {
        let (_tmp, db) = db();
        let err = db
            .write(|tx| {
                tx.execute("INSERT INTO workspaces (id, created_at) VALUES ('ws', 'yesterday')", [])?;
                Ok(())
            })
            .unwrap_err();
        assert!(err.to_string().contains("cannot store TEXT value in INTEGER column"), "{err}");
    }

    #[test]
    #[serial]
    fn test_backfill_adds_only_missing_projects() {
        let (_tmp, db) = db();
        let _clock = clock_at(1000.0);
        db.projects().add("ws", "p1").unwrap();
        let _clock = clock_at(2000.0);
        let added = db.projects().backfill("ws", ["p1", "p2", "p3", "p2"]).unwrap();
        assert_eq!(added, 2);
        assert_eq!(sorted(db.projects().ids("ws").unwrap()), ["p1", "p2", "p3"]);
        assert_eq!(created_at(&db, "p1"), 1000);
    }

    #[test]
    #[serial]
    fn test_backfill_never_prunes() {
        let (_tmp, db) = db();
        db.projects().add("ws", "gone").unwrap();
        db.projects().backfill("ws", ["p1"]).unwrap();
        assert_eq!(sorted(db.projects().ids("ws").unwrap()), ["gone", "p1"]);
    }

    #[test]
    #[serial]
    fn test_backfill_with_nothing_missing_does_not_write() {
        let (_tmp, db) = db();
        db.projects().add("ws", "p1").unwrap();
        // A write would wait on this one and fail its busy timeout.
        let held = WorkspaceDB::new(db.db_path()).unwrap();
        let (locked, wait_for_lock) = std::sync::mpsc::channel();
        let (release, wait_for_release) = std::sync::mpsc::channel::<()>();
        let holder = std::thread::spawn(move || {
            held.write(|_| {
                locked.send(()).unwrap();
                wait_for_release.recv().unwrap();
                Ok(())
            })
            .unwrap();
        });
        wait_for_lock.recv().unwrap();
        let started = std::time::Instant::now();
        assert_eq!(db.projects().backfill("ws", ["p1"]).unwrap(), 0);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        release.send(()).unwrap();
        holder.join().unwrap();
    }

    #[test]
    #[serial]
    fn test_backfill_of_a_new_workspace_records_it() {
        let (_tmp, db) = db();
        assert_eq!(db.projects().backfill("ws", ["p1"]).unwrap(), 1);
        assert_eq!(count(&db, "workspaces"), 1);
        assert_eq!(db.projects().backfill("ws", std::iter::empty()).unwrap(), 0);
    }
}
