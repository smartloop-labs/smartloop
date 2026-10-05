//! Port of `tests/unit/test_skill_create_e2e.py`.
//!
//! E2E tests: skill creation writes correct front matter and directory
//! structure.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::workspace::{Project, Skill};

use crate::shared::model_service_b::{FakeLLM, service_with_workspace};

/// The project's state dir — where skills are written.
///
/// Skills hang off `project_dir`, never a client-supplied path, so this only
/// has to hand out a temp state dir.  A `cwd` pointing somewhere else entirely
/// is set on the projects below to prove it is ignored.
struct SkillWorkspace {
    _tmp: tempfile::TempDir,
    state_dir: PathBuf,
}

fn skill_workspace() -> SkillWorkspace {
    let tmp = tempfile::tempdir().expect("tempdir");
    let state_dir = tmp.path().join("state").join("proj1");
    std::fs::create_dir_all(&state_dir).expect("state dir");
    SkillWorkspace { _tmp: tmp, state_dir }
}

fn make_project(state_dir: &Path, name: &str) -> Project {
    Project {
        id: "proj1".into(),
        project_dir: state_dir.to_string_lossy().into_owned(),
        current: true,
        name: Some(name.into()),
        cwd: Some("/somewhere/else/entirely".into()),
        ..Project::default()
    }
}

fn skill(name: &str, description: Option<&str>, content: &str) -> Skill {
    Skill {
        id: "abc123".into(),
        name: name.into(),
        description: description.map(str::to_string),
        content: content.into(),
        system: false,
    }
}

fn read(path: PathBuf) -> String {
    std::fs::read_to_string(path).expect("SKILL.md readable")
}

// ------------------------------------------------------------- write_skill_md

#[test]
fn test_creates_skills_dir() {
    let ws = skill_workspace();
    ModelService::write_skill_md(&make_project(&ws.state_dir, "my-project"), &skill("code-review", None, "Review the code carefully."))
        .unwrap();

    assert!(ws.state_dir.join("skills").is_dir());
}

#[test]
fn test_creates_skill_subdir() {
    let ws = skill_workspace();
    ModelService::write_skill_md(&make_project(&ws.state_dir, "my-project"), &skill("code-review", None, "Review the code carefully."))
        .unwrap();

    assert!(ws.state_dir.join("skills").join("code-review").is_dir());
}

#[test]
fn test_creates_skill_md_file() {
    let ws = skill_workspace();
    ModelService::write_skill_md(&make_project(&ws.state_dir, "my-project"), &skill("code-review", None, "Review the code carefully."))
        .unwrap();

    assert!(ws.state_dir.join("skills").join("code-review").join("SKILL.md").is_file());
}

#[test]
fn test_front_matter_has_name() {
    let ws = skill_workspace();
    ModelService::write_skill_md(&make_project(&ws.state_dir, "my-project"), &skill("code-review", None, "Review the code carefully."))
        .unwrap();

    let content = read(ws.state_dir.join("skills").join("code-review").join("SKILL.md"));
    assert!(content.contains("---"));
    assert!(content.contains("name: code-review"));
}

#[test]
fn test_front_matter_has_description() {
    let ws = skill_workspace();
    ModelService::write_skill_md(
        &make_project(&ws.state_dir, "my-project"),
        &skill("code-review", Some("Does a thorough code review."), "Review the code carefully."),
    )
    .unwrap();

    let content = read(ws.state_dir.join("skills").join("code-review").join("SKILL.md"));
    assert!(content.contains("description: Does a thorough code review."));
}

#[test]
fn test_front_matter_omits_description_when_none() {
    let ws = skill_workspace();
    ModelService::write_skill_md(&make_project(&ws.state_dir, "my-project"), &skill("code-review", None, "Review the code carefully."))
        .unwrap();

    let content = read(ws.state_dir.join("skills").join("code-review").join("SKILL.md"));
    assert!(!content.contains("description:"));
}

#[test]
fn test_content_written_after_front_matter() {
    let ws = skill_workspace();
    let body = "Review the code carefully.";
    ModelService::write_skill_md(&make_project(&ws.state_dir, "my-project"), &skill("code-review", None, body)).unwrap();

    let content = read(ws.state_dir.join("skills").join("code-review").join("SKILL.md"));
    // Front matter ends with --- then a blank line before the body.
    assert!(content.ends_with(&format!("---\n\n{body}")));
}

#[test]
fn test_front_matter_delimiters_present() {
    let ws = skill_workspace();
    ModelService::write_skill_md(&make_project(&ws.state_dir, "my-project"), &skill("greet", None, "Say hello.")).unwrap();

    let content = read(ws.state_dir.join("skills").join("greet").join("SKILL.md"));
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines[0], "---");
    let closing = lines.iter().skip(1).position(|line| *line == "---").map(|i| i + 1);
    assert!(closing.is_some_and(|i| i > 0));
}

// ------------------------------------------------- add_skill creates the file
//
// Integration-level: add_skill on ModelService writes SKILL.md with front
// matter.

/// A service with its state set up just enough, `_get_project` answering
/// with the test project.
fn make_service(state_dir: &Path) -> (Arc<ModelService>, Project) {
    let project = make_project(state_dir, "my-project");
    let svc = service_with_workspace(&state_dir.to_string_lossy(), vec![project.clone()], vec![]);
    let llm = FakeLLM::new("mock_llm");
    *llm.reply.write() = r#"{"name": "greet", "body": "Greet the user politely."}"#.into();
    *svc.llm.write() = Some(llm.dyn_llm());
    *svc.large_llm.write() = None;
    svc.large_model_attempted.store(true, std::sync::atomic::Ordering::SeqCst);

    let found = project.clone();
    let mut hooks = ModelServiceHooks::default();
    hooks.get_project = Some(Arc::new(move |_id: &str| Ok(found.clone())));
    svc.set_hooks(hooks);
    (svc, project)
}

#[test]
fn test_add_skill_creates_skills_dir() {
    let ws = skill_workspace();
    let (svc, _project) = make_service(&ws.state_dir);

    svc.add_skill("proj1", "Greet the user politely.", Some("greet"), Some("Greeter")).unwrap();

    assert!(ws.state_dir.join("skills").is_dir());
}

#[test]
fn test_add_skill_creates_skill_md() {
    let ws = skill_workspace();
    let (svc, _project) = make_service(&ws.state_dir);

    let result = svc.add_skill("proj1", "Greet the user politely.", Some("greet"), Some("Greeter")).unwrap();

    assert!(ws.state_dir.join("skills").join("greet").join("SKILL.md").is_file());
    assert_eq!(result["name"], Value::from("greet"));
}

#[test]
fn test_add_skill_front_matter_name() {
    let ws = skill_workspace();
    let (svc, _project) = make_service(&ws.state_dir);

    svc.add_skill("proj1", "Greet the user.", Some("greet"), Some("Friendly greeter")).unwrap();

    let content = read(ws.state_dir.join("skills").join("greet").join("SKILL.md"));
    assert!(content.contains("name: greet"));
    assert!(content.contains("description: Friendly greeter"));
}

#[test]
fn test_add_skill_sanitizes_name() {
    let ws = skill_workspace();
    let (svc, _project) = make_service(&ws.state_dir);

    let result = svc.add_skill("proj1", "Do stuff.", Some("My Cool Skill!"), Some("Does stuff")).unwrap();

    assert_eq!(result["name"], Value::from("my-cool-skill"));
    assert!(ws.state_dir.join("skills").join("my-cool-skill").join("SKILL.md").is_file());
}
