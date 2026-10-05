//! Port of `tests/unit/test_mcp_definitions.py`.
//!
//! A connection described by a file: what it says, and how it is launched.
//!
//! Dropping a `.toml` into `~/.smartloop/mcp.d/` is how a server is added
//! without the UI, so what the file can express is the whole feature.  Two
//! things matter here: the launcher belongs to the server (`uvx` is the
//! default for a bare package name, not an assumption about every server), and
//! one malformed drop-in must not take the working connections down with it.
//!
//! These files used to also carry crawl plans — declarative walks that
//! vectorized a server's pages ahead of time.  Crawling is gone; a definition
//! is a connection now, and nothing here indexes anything.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use serial_test::serial;

use smartloop::mcp::definition_loader::{
    definitions, definitions_shared, load_definitions, parse_definitions, reload_definitions,
    reload_definitions_shared, user_definition_paths,
};
use smartloop::mcp::definitions::{retired_fields, MCPDefinition};

use crate::shared::mcp::{capture_logs, EnvGuard};

const MINIMAL: &str = r#"
[[mcp]]
name = "acme"
package = "acme-mcp"
"#;

fn launch(cmd: &str, args: &[&str]) -> (Option<String>, Vec<String>) {
    (Some(cmd.to_string()), args.iter().map(|a| a.to_string()).collect())
}

/// `MCPDefinition(**fields)`.
fn definition(fields: Value) -> MCPDefinition {
    MCPDefinition::from_raw(fields.as_object().cloned().unwrap()).expect("a valid definition")
}

fn parse_one(text: &str, key: &str) -> MCPDefinition {
    parse_definitions(text, "test").expect("parses")[key].clone()
}

// ------------------------------------------------------------------ connections

#[test]
fn test_a_local_server_carries_its_command() {
    let text = r#"
[[mcp]]
name = "yahoo-finance"
package = "yahoo-finance-mcp"
"#;
    let d = parse_one(text, "yahoo-finance");
    assert_eq!(d.launch(), launch("uvx", &["yahoo-finance-mcp"]));
    assert_eq!(d.package.as_deref(), Some("yahoo-finance-mcp"));
    assert_eq!(d.kind(), Some("local"));
}

#[test]
fn test_a_remote_server_carries_its_url() {
    let text = r#"
[[mcp]]
name = "my-wiki"
url = "https://mcp.example.com/mcp"
"#;
    let d = parse_one(text, "my-wiki");
    assert_eq!(d.url.as_deref(), Some("https://mcp.example.com/mcp"));
    assert_eq!(d.kind(), Some("remote"));
}

#[test]
fn test_a_file_may_only_name_a_server() {
    // No connection at all still matches an installed one, which is how a file
    // dropped in later attaches to a server added through the UI.
    let d = parse_one("[[mcp]]\nname = \"notion\"\n", "notion");
    assert_eq!(d.kind(), None);
}

#[test]
fn test_the_name_becomes_the_key_and_a_match_term() {
    let text = r#"
[[mcp]]
name = "Yahoo-Finance"
match = ["yfinance"]
package = "yahoo-finance-mcp"
"#;
    let d = parse_one(text, "yahoo-finance");
    assert_eq!(d.key(), "yahoo-finance");
    assert!(d.match_terms().contains(&"yfinance".to_string()));
}

#[test]
fn test_a_misspelled_field_is_rejected() {
    // A typo must name the field rather than being silently ignored.
    let err = parse_definitions("[[mcp]]\nname = \"x\"\npackag = \"y\"\n", "test").unwrap_err();
    assert!(err.to_string().contains("packag"), "{err}");
}

// ------------------------------------------- files written for the old indexing format

/// A file that configured crawling still names a working connection.
///
/// Crawling is gone, so those fields describe nothing — but they sit beside a
/// url or a package name that is unaffected, and someone's existing file has
/// to keep installing rather than being rejected for saying something that is
/// now a no-op.
#[test]
fn test_the_retired_indexing_fields_are_ignored() {
    let text = r#"
[[mcp]]
name = "stocks"
package = "yahoo-finance-mcp"
crawl = true
index_every = "15m"

[[mcp.index]]
label = "stock-info"
tools = ["get_stock_info"]
max_pages = 20
max_parents = 25
"#;
    let d = parse_one(text, "stocks");
    assert_eq!(d.launch(), launch("uvx", &["yahoo-finance-mcp"]));
    assert_eq!(d.kind(), Some("local"));
    // `not hasattr(d, "index")`: the definition carries nothing of it.
    let serialized = serde_json::to_value(&d).unwrap();
    assert!(serialized.get("index").is_none());
}

#[test]
fn test_the_ignored_fields_are_named_in_the_log() {
    let text = "[[mcp]]\nname = \"x\"\nurl = \"https://e.example/mcp\"\nindex_every = \"6h\"\n";
    let (_, logged) = capture_logs(tracing::Level::WARN, || parse_definitions(text, "some-file.toml").unwrap());
    assert!(logged.contains("index_every"), "{logged}");
    assert!(logged.contains("some-file.toml"), "{logged}");
}

#[test]
fn test_a_typo_is_still_rejected_beside_them() {
    // Ignoring the retired keys must not turn into ignoring every unknown key:
    // `packag` still has to be reported.
    let text = "[[mcp]]\nname = \"x\"\nindex_every = \"6h\"\npackag = \"y\"\n";
    let err = parse_definitions(text, "test").unwrap_err();
    assert!(err.to_string().contains("packag"), "{err}");
}

#[test]
fn test_retired_fields_reports_what_an_entry_carries() {
    let raw = json!({"name": "x", "index_every": "1h", "index": []});
    assert_eq!(retired_fields(raw.as_object().unwrap()), vec!["index_every".to_string(), "index".to_string()]);
    let raw = json!({"name": "x", "package": "y"});
    assert!(retired_fields(raw.as_object().unwrap()).is_empty());
}

// --------------------------------------------------------------------- launchers
//
// `uvx` is the default for a bare name, not an assumption about all of them:
// most MCP servers ship on npm, and plenty are a module or a path.

#[test]
fn test_a_package_can_name_its_own_launcher() {
    let text = r#"
[[mcp]]
name = "node-server"
package = "npx -y @example/some-mcp"
"#;
    let d = parse_one(text, "node-server");
    assert_eq!(d.kind(), Some("local"));
    assert_eq!(d.launch(), launch("npx", &["-y", "@example/some-mcp"]));
}

#[test]
fn test_a_scoped_npm_name_is_run_with_npx() {
    // `@scope/pkg` cannot be a PyPI package, so uvx would only ever fail.
    let text = r#"
[[mcp]]
name = "filesystem"
package = "@modelcontextprotocol/server-filesystem"
"#;
    let d = parse_one(text, "filesystem");
    assert_eq!(d.launch(), launch("npx", &["-y", "@modelcontextprotocol/server-filesystem"]));
}

#[test]
fn test_a_module_or_a_path_is_a_command_line_too() {
    let text = r#"
[[mcp]]
name = "local-module"
package = "python -m my.server --port 0"
"#;
    let d = parse_one(text, "local-module");
    assert_eq!(d.launch(), launch("python", &["-m", "my.server", "--port", "0"]));
}

#[test]
fn test_a_command_is_there_for_what_a_package_name_cannot_say() {
    let text = r#"
[[mcp]]
name = "custom"
command = "npx"
args = ["-y", "some-mcp"]
"#;
    let d = parse_one(text, "custom");
    assert_eq!(d.launch(), launch("npx", &["-y", "some-mcp"]));
}

#[test]
fn test_a_command_may_carry_its_own_arguments() {
    let text = r#"
[[mcp]]
name = "inline"
command = "uv tool run --from pkg pkg"
"#;
    let d = parse_one(text, "inline");
    assert_eq!(d.launch(), launch("uv", &["tool", "run", "--from", "pkg", "pkg"]));
}

#[test]
#[serial]
fn test_a_command_expands_home_directory() {
    let _env = EnvGuard::set(&[("HOME", Some("/home/tester")), ("SNAP_NAME", None)]);
    let d = definition(json!({"name": "local", "command": "~/.local/craft mcp"}));
    assert_eq!(d.launch(), launch("/home/tester/.local/craft", &["mcp"]));
}

#[test]
#[serial]
fn test_a_tilde_reads_past_an_app_sandbox_container() {
    // Inside the macOS App Sandbox $HOME is the container, not the user's home,
    // so `~/.local/bin/craft` expanded to a path under .../Containers/<id>/Data
    // that nothing had ever installed into — the ENOENT every local server
    // added to the sandboxed build failed with.
    let _env = EnvGuard::set(&[("HOME", Some("/Users/tester/Library/Containers/ai.smartloop.studio/Data"))]);
    let d = definition(json!({"name": "local", "command": "~/.local/bin/craft mcp"}));
    // Python monkeypatches `smartloop.paths.real_home`; the seam here is the
    // injectable resolver.
    assert_eq!(
        d.launch_with_home(|| "/Users/tester".to_string()),
        launch("/Users/tester/.local/bin/craft", &["mcp"])
    );
}

#[test]
fn test_args_may_be_written_as_one_string() {
    // How everyone writes a command down, and how every other MCP client
    // displays one; iterating it per character is the alternative.
    let text = r#"
[[mcp]]
name = "stringy"
command = "npx"
args = "-y some-mcp"
"#;
    let d = parse_one(text, "stringy");
    assert_eq!(d.launch(), launch("npx", &["-y", "some-mcp"]));
}

#[test]
fn test_a_quoted_argument_survives_as_one_argument() {
    let d = definition(json!({"name": "q", "command": "python -c \"import x; x.main()\""}));
    assert_eq!(d.launch(), launch("python", &["-c", "import x; x.main()"]));
}

#[test]
fn test_args_without_a_command_is_an_error_naming_the_field() {
    let text = r#"
[[mcp]]
name = "orphan"
args = ["-y", "some-mcp"]
"#;
    let err = parse_definitions(text, "test").unwrap_err();
    assert!(err.to_string().contains("args without a command"), "{err}");
}

#[test]
fn test_a_package_and_a_command_say_the_same_thing_twice() {
    let text = r#"
[[mcp]]
name = "confused"
package = "x"
command = "npx"
"#;
    let err = parse_definitions(text, "test").unwrap_err();
    assert!(err.to_string().contains("twice"), "{err}");
}

#[test]
fn test_a_server_is_local_or_remote_not_both() {
    let text = r#"
[[mcp]]
name = "confused"
package = "x"
url = "https://x"
"#;
    let err = parse_definitions(text, "test").unwrap_err();
    assert!(err.to_string().contains("alternatives"), "{err}");
}

// -------------------------------------------------------------------- user files

fn write(home: &Path, name: &str, text: &str, in_dropin_dir: bool) -> PathBuf {
    let directory = if in_dropin_dir { home.join("mcp.d") } else { home.to_path_buf() };
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(name);
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn test_a_dropped_in_file_is_loaded() {
    let tmp = tempfile::tempdir().unwrap();
    write(tmp.path(), "acme.toml", MINIMAL, true);
    assert!(load_definitions(Some(tmp.path())).contains_key("acme"));
}

#[test]
fn test_a_single_user_file_also_works() {
    let tmp = tempfile::tempdir().unwrap();
    write(tmp.path(), "mcp.toml", MINIMAL, false);
    assert!(load_definitions(Some(tmp.path())).contains_key("acme"));
}

#[test]
fn test_a_malformed_drop_in_does_not_take_the_others_down() {
    let tmp = tempfile::tempdir().unwrap();
    write(tmp.path(), "aaa-broken.toml", "this is not [valid toml", true);
    write(tmp.path(), "bbb-good.toml", MINIMAL, true);
    assert!(load_definitions(Some(tmp.path())).contains_key("acme"));
}

#[test]
fn test_drop_ins_are_read_in_name_order() {
    let tmp = tempfile::tempdir().unwrap();
    write(tmp.path(), "10-first.toml", MINIMAL, true);
    write(
        tmp.path(),
        "20-second.toml",
        &MINIMAL.replace(r#"package = "acme-mcp""#, r#"package = "npx -y acme-mcp""#),
        true,
    );
    // Later file wins for the same key.
    let loaded = load_definitions(Some(tmp.path()))["acme"].clone();
    assert_eq!(loaded.launch(), launch("npx", &["-y", "acme-mcp"]));
}

#[test]
fn test_user_paths_list_the_single_file_before_the_directory() {
    let tmp = tempfile::tempdir().unwrap();
    write(tmp.path(), "mcp.toml", MINIMAL, false);
    write(tmp.path(), "z.toml", MINIMAL, true);
    let paths = user_definition_paths(Some(tmp.path()));
    assert!(paths[0].ends_with("mcp.toml"));
    assert!(paths[1].ends_with(Path::new("mcp.d").join("z.toml")));
}

#[test]
fn test_no_user_files_is_not_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(user_definition_paths(Some(tmp.path())).is_empty());
    assert!(load_definitions(Some(tmp.path())).is_empty());
}

// ------------------------------------------------------------------------- cache

#[test]
#[serial]
#[serial(mcp_global)]
fn test_the_registry_is_cached_and_reloadable() {
    // `definitions() is first`: the shared form is the cached registry itself.
    let first = definitions_shared(None);
    assert!(std::sync::Arc::ptr_eq(&definitions_shared(None), &first));
    assert!(!std::sync::Arc::ptr_eq(&reload_definitions_shared(), &first));
    // The owned forms read the same registry.
    assert_eq!(definitions(None), *reload_definitions_shared());
    assert_eq!(reload_definitions(), *definitions_shared(None));
}

// ------------------------------------------------------ finding the file for a server

/// Deleting a connection has to delete the file that defined it, or re-adding
/// a server by the same name silently inherits the old command.
mod finding_the_file_for_a_server {
    use std::sync::Arc;

    use smartloop::mcp::definition_loader::{set_definitions_override, DefinitionRegistry};
    use smartloop::mcp::definitions::definition_for;
    use smartloop::workspace::MCPServer;

    use super::*;

    fn server(name: &str, catalog_slug: Option<&str>) -> MCPServer {
        MCPServer {
            name: name.to_string(),
            catalog_slug: catalog_slug.map(str::to_string),
            pkg: None,
            remote_config: None,
            config: None,
            ..Default::default()
        }
    }

    /// `monkeypatch.setattr("...definition_loader.definitions", lambda: found)`
    /// for the duration of `f`.
    fn with_definitions<R>(found: DefinitionRegistry, f: impl FnOnce() -> R) -> R {
        let found = Arc::new(found);
        set_definitions_override(Some(Arc::new(move || found.clone())));
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        set_definitions_override(None);
        out.unwrap_or_else(|p| std::panic::resume_unwind(p))
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_the_catalog_slug_matches_first() {
        let found = parse_definitions(MINIMAL, "test").unwrap();
        let expected = found["acme"].clone();
        let got = with_definitions(found, || definition_for(&server("Something Else", Some("acme"))));
        assert_eq!(got, Some(expected));
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_a_declared_match_term_finds_it() {
        let found =
            parse_definitions("[[mcp]]\nname = \"acme\"\nmatch = [\"yfinance\"]\npackage = \"x\"\n", "test").unwrap();
        let expected = found["acme"].clone();
        let got = with_definitions(found, || definition_for(&server("YFinance tools", None)));
        assert_eq!(got, Some(expected));
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_an_unrelated_server_matches_nothing() {
        let found = parse_definitions(MINIMAL, "test").unwrap();
        let got = with_definitions(found, || definition_for(&server("notion", None)));
        assert_eq!(got, None);
    }
}

// ----------------------------------------------------- nothing rewrites the command

/// What a connection configures is what runs.
///
/// Registration used to fold `uvx <pkg>` into `uvx --from <pkg> <pkg>` and
/// swap `uvx` for `uv tool run` — so the command in the config was not the
/// command that ran, invisibly, and only ever correctly for uv.
mod nothing_rewrites_the_command {
    use smartloop::workspace::workspace::MCPServerConfig;

    use super::*;
    use crate::shared::mcp::ServiceEnv;

    /// Register through the real service and read back the stored config.
    /// (Python mocks `_get_project` and patches `LocalMCPClient`; registration
    /// itself never launches anything — discovery runs on the MCP task.)
    fn register(command: &str, args: &[&str]) -> MCPServerConfig {
        let env = ServiceEnv::new("p1");
        env.model_service
            .add_local_mcp_server(
                command,
                Some(args.iter().map(|a| a.to_string()).collect()),
                None,
                None,
                None,
                None,
                None,
                None,
                Some("p1"),
            )
            .expect("registered");
        env.servers()[0].config.clone().expect("a local config")
    }

    fn stored(config: &MCPServerConfig) -> (String, Vec<String>) {
        (config.command.clone(), config.args.clone())
    }

    fn expected(cmd: &str, args: &[&str]) -> (String, Vec<String>) {
        (cmd.to_string(), args.iter().map(|a| a.to_string()).collect())
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_a_bare_uvx_package_is_stored_as_written() {
        let config = register("uvx", &["yahoo-finance-mcp"]);
        assert_eq!(stored(&config), expected("uvx", &["yahoo-finance-mcp"]));
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_an_npx_invocation_is_stored_as_written() {
        let config = register("npx", &["-y", "@example/some-mcp"]);
        assert_eq!(stored(&config), expected("npx", &["-y", "@example/some-mcp"]));
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_a_command_line_in_one_string_is_only_split() {
        let config = register("uv tool run --from pkg pkg", &[]);
        assert_eq!(stored(&config), expected("uv", &["tool", "run", "--from", "pkg", "pkg"]));
    }

    #[test]
    #[serial]
    #[serial(mcp_global)]
    fn test_a_program_path_needs_no_arguments() {
        let config = register("/opt/bin/my-mcp", &[]);
        assert_eq!(stored(&config), expected("/opt/bin/my-mcp", &[]));
    }
}

// --------------------------------------------------- the command is resolved to its path

/// Which `uvx` — the app's PATH is not the shell's, so a command a user
/// verified in a terminal can be missing from the process that launches it.
mod the_command_is_resolved_to_its_path {
    use std::collections::HashMap;

    use smartloop::mcp::local_client::LocalMCPClient;

    use super::*;

    fn client(command: &str, args: &[&str]) -> LocalMCPClient {
        LocalMCPClient::new(command, args.iter().map(|a| a.to_string()).collect())
    }

    fn default_args() -> &'static [&'static str] {
        &["--from", "pkg", "pkg"]
    }

    fn env(path: Option<&str>) -> HashMap<String, String> {
        path.map(|p| HashMap::from([("PATH".to_string(), p.to_string())])).unwrap_or_default()
    }

    #[test]
    fn test_a_command_on_the_path_resolves_to_its_full_path() {
        let tmp = tempfile::tempdir().unwrap();
        let program = tmp.path().join("uvx");
        std::fs::write(&program, "#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let resolved = client("uvx", default_args()).resolve_command(&env(Some(&tmp.path().to_string_lossy())));

        assert_eq!(resolved, program.to_string_lossy());
    }

    #[test]
    #[serial]
    fn test_an_absolute_path_is_used_as_given() {
        // Not inside the App Sandbox (where a bundled copy would be preferred).
        let _env = EnvGuard::set(&[("HOME", Some("/Users/tester")), ("SNAP_NAME", None)]);
        assert_eq!(client("/opt/bin/my-mcp", &[]).resolve_command(&env(None)), "/opt/bin/my-mcp");
    }

    #[test]
    fn test_a_command_nowhere_on_the_path_is_left_to_the_os() {
        // The OS produces the error rather than this inventing one.
        assert_eq!(
            client("definitely-not-installed", default_args()).resolve_command(&env(Some("/nonexistent"))),
            "definitely-not-installed"
        );
    }

    #[test]
    #[serial]
    fn test_a_built_in_server_runs_under_the_current_runtime() {
        // `sys.executable` is the running binary (`SLP_EXE` overrides it).
        let _env = EnvGuard::set(&[("SLP_EXE", None)]);
        let c = client("python", &["-m", "smartloop.api.mcp.system_server"]);
        let current = std::env::current_exe().unwrap().to_string_lossy().into_owned();
        assert_eq!(c.resolve_command(&env(None)), current);
    }
}
