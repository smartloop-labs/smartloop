//! Port of `tests/unit/test_mcp_definition_install.py`.
//!
//! Installing a connection from a dropped-in TOML file.
//!
//! The file is the whole definition — what the server is called and how to
//! reach it.  Two things have to land: the file on disk under `mcp.d` so the
//! connection can be edited or removed as a file, and the connection itself
//! through the ordinary registration path so an OAuth server behaves exactly
//! as a typed-in URL does.
//!
//! Most of what's pinned here is the file being rejected, because a dropped
//! file is the one input nobody proofreads.

use std::path::Path;
use std::sync::{Arc, Mutex};

use smartloop::api::model::mcp::{MCPDefinitionRequest, MCPRegisterRequest, MCPRegisterResponse};
use smartloop::api::routers::mcp_servers::{register_mcp_from_definition_with, HTTPException};

const LOCAL: &str = r#"
[[mcp]]
name = "yahoo-finance"
package = "yahoo-finance-mcp"
env = { TZ = "UTC" }
"#;

const REMOTE: &str = r#"
[[mcp]]
name = "my-wiki"
url = "https://mcp.example.com/mcp"
"#;

/// Points `SLP_HOME` at the test's home for the duration (the Python test
/// patches `AppSettings().home_dir`), so the definition reload after saving
/// reads the test's files rather than the real home.
struct HomeGuard {
    previous: Option<String>,
}

impl HomeGuard {
    fn set(home: &Path) -> Self {
        let previous = std::env::var("SLP_HOME").ok();
        // SAFETY: the tests touching SLP_HOME are #[serial].
        unsafe { std::env::set_var("SLP_HOME", home) };
        Self { previous }
    }
}

impl Drop for HomeGuard {
    fn drop(&mut self) {
        // SAFETY: the tests touching SLP_HOME are #[serial].
        unsafe {
            match &self.previous {
                Some(v) => std::env::set_var("SLP_HOME", v),
                None => std::env::remove_var("SLP_HOME"),
            }
        }
    }
}

/// Run the endpoint with registration stubbed, returning (result, request).
async fn install_named(
    toml: &str,
    home: &Path,
    filename: &str,
) -> (Result<MCPRegisterResponse, HTTPException>, Option<MCPRegisterRequest>) {
    let _home = HomeGuard::set(home);
    let registered: Arc<Mutex<Option<MCPRegisterRequest>>> = Arc::new(Mutex::new(None));
    let seen = registered.clone();
    let result = register_mcp_from_definition_with(
        "p1",
        MCPDefinitionRequest { toml: toml.to_string(), filename: Some(filename.to_string()) },
        home,
        move |_project_id, request| async move {
            *seen.lock().unwrap() = Some(request);
            Ok(MCPRegisterResponse { auth_type: "none".to_string(), ..Default::default() })
        },
    )
    .await;
    let request = registered.lock().unwrap().take();
    (result, request)
}

async fn install(
    toml: &str,
    home: &Path,
) -> (Result<MCPRegisterResponse, HTTPException>, Option<MCPRegisterRequest>) {
    install_named(toml, home, "dropped.toml").await
}

fn listdir(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

mod test_install {
    use super::*;
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_a_local_definition_registers_its_command() {
        let tmp = tempfile::tempdir().unwrap();
        let (_, request) = install(LOCAL, tmp.path()).await;
        let request = request.expect("registration was called");
        assert_eq!(request.server_type, "local");
        assert_eq!(request.command.as_deref(), Some("uvx"));
        assert_eq!(request.args, Some(vec!["yahoo-finance-mcp".to_string()]));
        assert_eq!(request.name.as_deref(), Some("yahoo-finance"));
    }

    #[tokio::test]
    #[serial]
    async fn test_a_remote_definition_registers_its_url() {
        let tmp = tempfile::tempdir().unwrap();
        let (_, request) = install(REMOTE, tmp.path()).await;
        let request = request.expect("registration was called");
        assert_eq!(request.server_type, "remote");
        assert_eq!(request.server_url.as_deref(), Some("https://mcp.example.com/mcp"));
        // So an OAuth server can hand back to the app, same as a typed URL.
        assert_eq!(request.return_url.as_deref(), Some("smartloop://mcp-authorization"));
    }

    #[tokio::test]
    #[serial]
    async fn test_the_file_is_on_disk_before_registration_returns() {
        // Written before the registration it describes, so a failed
        // registration leaves something to retry from rather than nothing.
        let tmp = tempfile::tempdir().unwrap();
        let _ = install(LOCAL, tmp.path()).await;
        let saved = tmp.path().join("mcp.d").join("yahoo-finance.toml");
        assert!(saved.is_file());
        assert!(std::fs::read_to_string(&saved).unwrap().contains("yahoo-finance-mcp"));
    }

    #[tokio::test]
    #[serial]
    async fn test_the_saved_file_is_named_after_the_connection() {
        // Not after whatever it was called on the way in — re-dropping an
        // edited file has to replace it, not stack a second copy beside it.
        let tmp = tempfile::tempdir().unwrap();
        let _ = install_named(LOCAL, tmp.path(), "my-download (3).toml").await;
        assert_eq!(listdir(&tmp.path().join("mcp.d")), vec!["yahoo-finance.toml".to_string()]);
    }

    #[tokio::test]
    #[serial]
    async fn test_re_dropping_an_edited_file_replaces_it() {
        let tmp = tempfile::tempdir().unwrap();
        let _ = install(LOCAL, tmp.path()).await;
        let edited = LOCAL.replace("TZ = \"UTC\"", "TZ = \"Europe/Berlin\"");
        let _ = install(&edited, tmp.path()).await;
        let saved =
            std::fs::read_to_string(tmp.path().join("mcp.d").join("yahoo-finance.toml")).unwrap();
        assert!(saved.contains("Europe/Berlin") && !saved.contains("UTC"));
        assert_eq!(listdir(&tmp.path().join("mcp.d")), vec!["yahoo-finance.toml".to_string()]);
    }
}

mod test_rejections {
    use super::*;
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_a_file_that_is_not_toml_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let (result, _) = install("this is not [valid toml", tmp.path()).await;
        let e = result.expect_err("an invalid file must be refused");
        assert_eq!(e.status_code, 400);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_file_with_no_mcp_entry_says_so() {
        let tmp = tempfile::tempdir().unwrap();
        let (result, _) = install("[something_else]\nfoo = \"bar\"\n", tmp.path()).await;
        let e = result.expect_err("a file with no [[mcp]] must be refused");
        assert!(e.detail.contains("[[mcp]]"), "{}", e.detail);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_definition_with_no_connection_is_refused() {
        // A name alone can't be installed — there is nothing to install.
        let toml = "\n[[mcp]]\nname = \"orphan\"\n";
        let tmp = tempfile::tempdir().unwrap();
        let (result, _) = install(toml, tmp.path()).await;
        let e = result.expect_err("a definition with no connection must be refused");
        assert!(e.detail.contains("no connection"), "{}", e.detail);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_file_describing_several_servers_is_refused() {
        // One file per connection: the file is saved under the connection's
        // name, and two connections have no single name to save it under.
        let tmp = tempfile::tempdir().unwrap();
        let both = format!("{LOCAL}{REMOTE}");
        let (result, _) = install(&both, tmp.path()).await;
        let e = result.expect_err("several connections must be refused");
        assert!(e.detail.contains("one file per connection"), "{}", e.detail);
    }

    #[tokio::test]
    #[serial]
    async fn test_a_misspelled_field_fails_the_whole_file() {
        // Silently ignoring it would install a connection missing the thing
        // the author wrote the file to set.
        let toml = "\n[[mcp]]\nname = \"acme\"\npackag = \"acme-mcp\"\n";
        let tmp = tempfile::tempdir().unwrap();
        let (result, _) = install(toml, tmp.path()).await;
        let e = result.expect_err("a misspelled field must fail the file");
        assert_eq!(e.status_code, 400);
        assert!(!tmp.path().join("mcp.d").exists());
    }

    #[tokio::test]
    #[serial]
    async fn test_an_unwritable_drop_in_dir_still_installs_the_connection() {
        // Keeping the file is a bonus; the connection is the point, and losing
        // the first must not cost the second.
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        std::fs::create_dir(&home).unwrap();
        std::fs::write(home.join("mcp.d"), "not a directory").unwrap();
        let (_, request) = install(LOCAL, &home).await;
        let request = request.expect("registration still ran");
        assert_eq!(request.command.as_deref(), Some("uvx"));
    }
}
