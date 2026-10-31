//! Auto-write: an answer that lays out files is written as those files in the
//! working directory; existing files and paths outside it are left alone.

use smartloop::api::routers::chat::write_summary;
use smartloop::api::services::output_writer::{
    extract_file_blocks, resolve_under, write_document, write_file_blocks, WriteOutcome,
};

const PROJECT_ANSWER: &str = "Here's a minimal Rust project.

File: Cargo.toml
```toml
[package]
name = \"hello\"
version = \"0.1.0\"
```

Create `src/main.rs`:

```rust
fn main() {
    println!(\"Hello, world!\");
}
```

### .gitignore
```
/target
```

Run it with:
```bash
cargo run
```
";

#[test]
fn test_extract_file_blocks_reads_labelled_blocks_only() {
    let blocks = extract_file_blocks(PROJECT_ANSWER);
    let paths: Vec<&str> = blocks.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(paths, vec!["Cargo.toml", "src/main.rs", ".gitignore"]);
    assert_eq!(blocks[1].1, "fn main() {\n    println!(\"Hello, world!\");\n}");
}

#[test]
fn test_extract_file_blocks_path_on_fence_info() {
    let blocks = extract_file_blocks("```rust:src/lib.rs\npub fn f() {}\n```\n\n```py app/main.py\nprint(1)\n```");
    assert_eq!(
        blocks,
        vec![
            ("src/lib.rs".to_string(), "pub fn f() {}".to_string()),
            ("app/main.py".to_string(), "print(1)".to_string()),
        ]
    );
}

#[test]
fn test_extract_file_blocks_keeps_nested_fences() {
    let text = "File: README.md\n````markdown\n# Demo\n\n```bash\ncargo run\n```\n````\n";
    let blocks = extract_file_blocks(text);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].1, "# Demo\n\n```bash\ncargo run\n```");
}

#[test]
fn test_extract_file_blocks_ignores_prose_and_unlabelled_code() {
    for text in [
        "Install it:\n```bash\ncargo new hello\n```",
        "# A document\n\nNo code here.",
        "Use e.g. this:\n```\nx\n```",
    ] {
        assert!(extract_file_blocks(text).is_empty(), "unexpected blocks: {text}");
    }
}

#[test]
fn test_extract_file_blocks_later_block_wins() {
    let text = "File: a.txt\n```\none\n```\nFile: a.txt\n```\ntwo\n```";
    assert_eq!(extract_file_blocks(text), vec![("a.txt".to_string(), "two".to_string())]);
}

#[test]
fn test_resolve_under_rejects_escapes() {
    let root = std::path::Path::new("/work");
    assert_eq!(resolve_under(root, "./src/main.rs"), Some(root.join("src/main.rs")));
    assert_eq!(resolve_under(root, "../etc/passwd"), None);
    assert_eq!(resolve_under(root, "src/../../x"), None);
    assert_eq!(resolve_under(root, "/etc/passwd"), None);
}

#[test]
fn test_write_file_blocks_creates_files_and_keeps_existing() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".gitignore"), "mine\n").unwrap();

    let outcome = write_file_blocks(dir.path(), &extract_file_blocks(PROJECT_ANSWER)).unwrap();

    assert_eq!(outcome.written.len(), 2);
    assert_eq!(outcome.skipped, vec![dir.path().join(".gitignore").to_string_lossy().into_owned()]);
    let main = std::fs::read_to_string(dir.path().join("src/main.rs")).unwrap();
    assert!(main.starts_with("fn main()") && main.ends_with("}\n"));
    assert_eq!(std::fs::read_to_string(dir.path().join(".gitignore")).unwrap(), "mine\n");
}

#[test]
fn test_write_file_blocks_skips_paths_outside_root() {
    let dir = tempfile::tempdir().unwrap();
    let blocks = vec![("../escape.txt".to_string(), "x".to_string())];
    assert_eq!(write_file_blocks(dir.path(), &blocks).unwrap(), WriteOutcome::default());
    assert!(!dir.path().parent().unwrap().join("escape.txt").exists());
}

/// A saved answer goes under `.out/`, which the folder's index and sync
/// leave alone, so it never comes back as a document of the project.
#[test]
fn test_write_document_goes_in_out_without_replacing() {
    let dir = tempfile::tempdir().unwrap();
    let first = write_document(dir.path(), "notes", "# One").unwrap();
    let second = write_document(dir.path(), "notes", "# Two").unwrap();
    assert_eq!(first, dir.path().join(".out/notes.md").to_string_lossy());
    assert_eq!(second, dir.path().join(".out/notes_1.md").to_string_lossy());
    assert!(!dir.path().join("notes.md").exists());
}

#[test]
fn test_write_summary_links_files_and_notes_skipped() {
    let single = WriteOutcome { written: vec!["/w/notes.md".into()], skipped: vec![] };
    assert_eq!(write_summary(&single), "\n\n[notes.md](/w/notes.md)");

    let many = WriteOutcome {
        written: vec!["/w/Cargo.toml".into(), "/w/src/main.rs".into()],
        skipped: vec!["/w/.gitignore".into()],
    };
    assert_eq!(
        write_summary(&many),
        "\n\nWrote 2 files:\n\n- [Cargo.toml](/w/Cargo.toml)\n- [main.rs](/w/src/main.rs)\
         \n\nLeft unchanged (already exist): `/w/.gitignore`"
    );
    assert_eq!(write_summary(&WriteOutcome::default()), "");
}
