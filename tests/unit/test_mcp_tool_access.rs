//! What a tool does, as collect decides whether to run it unprompted.
//!
//! The server's annotation decides when it gives one; the verbs in the name
//! cover tools without annotations.  Names below are real tools from the
//! Linear, Notion, Trello and Clera servers.

use smartloop::agents::mcp::explorer::ToolAccess;
use smartloop::workspace::{MCPServer, MCPToolAnnotations, MCPToolDefinition};

fn annotated(name: &str, read_only: Option<bool>, destructive: Option<bool>) -> MCPToolDefinition {
    MCPToolDefinition {
        name: name.to_string(),
        annotations: Some(MCPToolAnnotations {
            read_only_hint: read_only,
            destructive_hint: destructive,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn bare(name: &str) -> MCPToolDefinition {
    MCPToolDefinition { name: name.to_string(), ..Default::default() }
}

#[test]
fn test_names_without_annotations() {
    for (name, expected) in [
        ("list_issues", ToolAccess::Read),
        ("notion-search", ToolAccess::Read),
        ("trelloReadCard", ToolAccess::Read),
        ("notion-download-attachment", ToolAccess::Read),
        ("save_issue", ToolAccess::Write),
        ("notion-move-pages", ToolAccess::Write),
        ("notion-create-pages", ToolAccess::Write),
        ("upload_file", ToolAccess::Write),
        // A write word anywhere outranks a read word.
        ("mark_as_read", ToolAccess::Write),
        ("whoami", ToolAccess::Unknown),
        ("extract_images", ToolAccess::Unknown),
        ("", ToolAccess::Unknown),
    ] {
        assert_eq!(ToolAccess::of(&bare(name)), expected, "{name}");
    }
}

/// Clera's `whoami` has no verb in its name; the server says it reads.
#[test]
fn test_a_read_only_annotation_wins_over_the_name() {
    assert_eq!(ToolAccess::of(&annotated("whoami", Some(true), Some(false))), ToolAccess::Read);
    assert_eq!(ToolAccess::of(&annotated("notion-get-suggested-edit", Some(true), None)), ToolAccess::Read);
}

/// Clera's `getMoreTools` records feedback; the name alone reads like a lookup.
#[test]
fn test_a_not_read_only_annotation_wins_over_the_name() {
    assert_eq!(ToolAccess::of(&annotated("getMoreTools", Some(false), Some(false))), ToolAccess::Write);
    // Linear marks its own whoami as not read-only, and that is believed too.
    assert!(!ToolAccess::of(&annotated("whoami", Some(false), None)).is_read());
}

/// The spec treats a tool that is not read-only as destructive unless the
/// server says otherwise.
#[test]
fn test_destructive_follows_the_spec_default() {
    assert_eq!(ToolAccess::of(&annotated("delete_issue", Some(false), None)), ToolAccess::Destructive);
    assert_eq!(ToolAccess::of(&annotated("delete_issue", Some(false), Some(true))), ToolAccess::Destructive);
    assert_eq!(ToolAccess::of(&annotated("delete_issue", None, Some(true))), ToolAccess::Destructive);
}

/// Annotations with no read-only or destructive hint leave it to the name.
#[test]
fn test_other_hints_alone_fall_back_to_the_name() {
    let only_idempotent = MCPToolDefinition {
        name: "save_issue".into(),
        annotations: Some(MCPToolAnnotations { idempotent_hint: Some(true), ..Default::default() }),
        ..Default::default()
    };
    assert_eq!(ToolAccess::of(&only_idempotent), ToolAccess::Write);
    assert_eq!(ToolAccess::of(&annotated("list_issues", None, Some(false))), ToolAccess::Read);
}

#[test]
fn test_only_a_read_is_read() {
    assert!(ToolAccess::Read.is_read());
    assert!(!ToolAccess::Write.is_read());
    assert!(!ToolAccess::Destructive.is_read());
    assert!(!ToolAccess::Unknown.is_read());
}

#[test]
fn test_a_tool_missing_from_the_saved_list_is_judged_by_name() {
    let server = MCPServer { tools: vec![annotated("whoami", Some(true), None)], ..Default::default() };

    assert_eq!(ToolAccess::on(&server, "whoami"), ToolAccess::Read);
    assert_eq!(ToolAccess::on(&server, "save_issue"), ToolAccess::Write);
    assert_eq!(ToolAccess::on(&server, "whoami_v2"), ToolAccess::Unknown);
}
