//! MCP Skills server – `save-skill`, `update-skill`, `delete-skill`,
//! `list-skills`, `read-skill`.
//!
//! A skill is a reusable, composite user prompt.  This server lets the model
//! save one the way it calls any other tool: the user says "save this as a
//! skill" (or "generate a skill from this conversation"), the model drafts
//! the skill from the conversation it can see, and calls `save-skill`.  "Add
//! a cool-down to my morning skill" and "delete the morning skill" work the
//! same way.  `list-skills` and `read-skill` are read-only, so the collect pass
//! looks a skill up before a planned update step rewrites it.
//!
//! The skill is written as `<skills dir>/<name>/SKILL.md`, exactly as the
//! `POST /v1/projects/{id}/skills` endpoint writes it (skills are read from
//! disk, so it is matched on the next request).  The project's skills
//! directory comes from the `SMARTLOOP_SKILLS_DIR` environment variable, set
//! when the server is registered on the project.
//!
//! Served over stdio by the hidden `slp __skill-server` subcommand.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock, Implementation, JsonObject, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, ServiceExt};
use serde_json::{json, Value};

use crate::api::services::model_service::ModelService;

/// MCP server name.
pub const SERVER_NAME: &str = "skills";

/// Id of the Skills server on a project.
pub const SKILLS_SERVER_ID: &str = "__skills__";

/// Display name of the Skills server on a project.
pub const SKILLS_SERVER_NAME: &str = "Skills";

/// Environment variable naming the project's skills directory.
pub const SKILLS_DIR_ENV: &str = "SMARTLOOP_SKILLS_DIR";

/// Tool names.
pub const SAVE_SKILL_TOOL: &str = "save-skill";
pub const UPDATE_SKILL_TOOL: &str = "update-skill";
pub const DELETE_SKILL_TOOL: &str = "delete-skill";
pub const LIST_SKILLS_TOOL: &str = "list-skills";
pub const READ_SKILL_TOOL: &str = "read-skill";

/// Where `delete-skill` moves a skill, under the skills directory.  Not a
/// skill itself (no `SKILL.md` directly inside), so the loader skips it.
pub const DELETED_DIR: &str = ".deleted";

pub const UPDATE_SKILL_DESCRIPTION: &str = "Change an existing skill in the current project. Call this when \
the user asks to update, edit, change or improve a skill (\"add a cool-down to my morning skill\", \"make the \
linear skill only look at my team\"). Call it directly — no need to read the skill first: the current text of \
the project's skills is given to you when you make this call. `name` is the skill's existing name. `content` \
replaces the whole body: start from the current text, apply the user's change, keep everything else. Follow the \
same structure and rules as save-skill: Goal, Inputs, Steps naming servers/tools, Output, Notes; self-contained. \
Omit `content` or `description` to keep it as it is.";

pub const DELETE_SKILL_DESCRIPTION: &str = "Delete a skill from the current project. Call this only when the \
user clearly asks to delete or remove a skill. `name` must be the skill's exact name (see list-skills).";

pub const LIST_SKILLS_DESCRIPTION: &str = "List the skills saved in the current project, with what each is for. \
Use to find a skill's exact name, or when the user asks which skills they have.";

pub const READ_SKILL_DESCRIPTION: &str = "Show a skill's current instructions, when the user asks what a skill \
says or does. To change a skill, call update-skill instead.";

/// Description shown to the model.  It does two jobs: it makes "save this as
/// a skill" resolve to this tool, and it tells the model how to write a skill
/// that works when it is matched and run later, from a fresh conversation.
pub const SAVE_SKILL_DESCRIPTION: &str = "Save a reusable skill to the current project. A skill is a \
reusable prompt: the instructions that let a later conversation repeat a task done in this one, without \
the user explaining it again. Call this when the user asks to save, create, generate or make a skill \
(\"save this as a skill\", \"turn this into a skill\", \"remember how to do this\").

Before calling, read back through the conversation and work out:
- the task the user wanted done, in general terms (not just the last message);
- which connected tools / MCP servers were used, for what, and in what order (e.g. Linear to list \
issues, GitHub to open a PR);
- the inputs that change each time (dates, ids, names) versus the ones that stay fixed (a team, a \
repo, a label) — keep a value fixed only if the user did;
- conditions and decisions made along the way (\"only if there are overdue issues, comment on them\");
- corrections the user made, and the format of the answer they accepted.

The skill is matched by its description and run later with only its content as guidance, so the \
content must stand on its own: no \"as above\", \"this conversation\" or \"the issue we discussed\".";

/// The tool's input schema; each field says what a good value looks like.
pub fn save_skill_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Short kebab-case name for the task, 2-4 words, e.g. `linear-weekly-triage`. \
                    Use the name the user gave, if any."
            },
            "description": {
                "type": "string",
                "description": "One sentence starting with 'Use when the user ...' that says which requests \
                    this skill is for — it is what future requests are matched against, so name the task, the \
                    service and typical phrasings. E.g. 'Use when the user asks to triage, review or summarize \
                    their open Linear issues for the week.'"
            },
            "content": {
                "type": "string",
                "description": "The skill's instructions in markdown, written as a prompt to follow next time. \
                    Use this structure:\n\
                    # <Title>\n\
                    <One or two sentences: the goal.>\n\
                    ## Inputs\n\
                    - <what to take from the user's request, with defaults, e.g. 'team — default DEV'>\n\
                    ## Steps\n\
                    1. <action> using <server / tool>, e.g. 'Pull the latest issues from the Linear MCP \
                    (list_issues), assigned to me, updated in the last 7 days.'\n\
                    2. <next action; say how it uses the previous step's result>\n\
                    3. <conditional step: 'If any issue is overdue, post a comment on it asking for an update.'>\n\
                    ## Output\n\
                    <the shape of the answer: a table, a summary with links, a short list...>\n\
                    ## Notes\n\
                    - <preferences and corrections the user gave, things to avoid>\n\
                    Be specific about servers, tools, filters and order; leave out anything that only \
                    applied to this one conversation."
            },
        },
        "required": ["name", "description", "content"],
    })
}

/// A skill-name argument.
fn name_property() -> Value {
    json!({"type": "string", "description": "The skill's name, e.g. good-morning-workout"})
}

/// Every tool's `(name, description, input schema)`, for the server and for
/// the project's record of the server.
pub fn tool_specs() -> Vec<(&'static str, &'static str, Value)> {
    let save = save_skill_schema();
    let field = |key: &str| save["properties"][key].clone();
    vec![
        (SAVE_SKILL_TOOL, SAVE_SKILL_DESCRIPTION, save.clone()),
        (
            UPDATE_SKILL_TOOL,
            UPDATE_SKILL_DESCRIPTION,
            json!({
                "type": "object",
                "properties": {
                    "name": name_property(),
                    "description": field("description"),
                    "content": field("content"),
                },
                "required": ["name"],
            }),
        ),
        (
            DELETE_SKILL_TOOL,
            DELETE_SKILL_DESCRIPTION,
            json!({"type": "object", "properties": {"name": name_property()}, "required": ["name"]}),
        ),
        (LIST_SKILLS_TOOL, LIST_SKILLS_DESCRIPTION, json!({"type": "object", "properties": {}})),
        (
            READ_SKILL_TOOL,
            READ_SKILL_DESCRIPTION,
            json!({"type": "object", "properties": {"name": name_property()}, "required": ["name"]}),
        ),
    ]
}

fn tool_definitions() -> Vec<Tool> {
    tool_specs()
        .into_iter()
        .map(|(name, description, schema)| {
            let schema = match schema {
                Value::Object(map) => Arc::new(map),
                _ => Arc::new(JsonObject::new()),
            };
            Tool::new(name, description, schema)
        })
        .collect()
}

fn optional_str<'a>(args: &'a JsonObject, field: &str) -> anyhow::Result<Option<&'a str>> {
    match args.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => anyhow::bail!("`{field}` must be a string"),
    }
}

fn required_str<'a>(args: &'a JsonObject, field: &str) -> anyhow::Result<&'a str> {
    match args.get(field) {
        Some(Value::String(s)) if !s.trim().is_empty() => Ok(s.as_str()),
        Some(Value::String(_)) => anyhow::bail!("`{field}` must not be empty"),
        None => anyhow::bail!("`{field}` is required"),
        Some(_) => anyhow::bail!("`{field}` must be a string"),
    }
}

/// Save a skill under `skills_dir`.  Refuses to replace an existing skill: a
/// tool call must never silently overwrite one the user wrote.
pub fn save_skill(skills_dir: &Path, name: &str, description: &str, content: &str) -> anyhow::Result<String> {
    if ModelService::is_system_skill(name.trim()) {
        anyhow::bail!("'{}' is a reserved skill name — choose a different name", name.trim());
    }
    let skill = ModelService::build_skill(content, Some(name), Some(description));
    if skills_dir.join(&skill.name).join("SKILL.md").exists() {
        anyhow::bail!(
            "a skill named '{}' already exists in this project — choose a different name",
            skill.name
        );
    }
    if ModelService::is_system_skill(&skill.name) {
        anyhow::bail!("'{}' is a reserved skill name — choose a different name", skill.name);
    }
    ModelService::write_skill_md_in(skills_dir, &skill)?;
    Ok(format!(
        "Saved skill '{}'. It will be used for matching requests from now on.\n\n{}",
        skill.name,
        ModelService::render_skill_content(&skill)
    ))
}

/// The user skills under `skills_dir`: `(name, description)`, by name.
pub fn list_skills(skills_dir: &Path) -> Vec<(String, Option<String>)> {
    let Ok(entries) = std::fs::read_dir(skills_dir) else { return Vec::new() };
    let mut out: Vec<(String, Option<String>)> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !name.starts_with('.') && !ModelService::is_system_skill(name))
        .filter_map(|name| {
            let raw = std::fs::read_to_string(skills_dir.join(&name).join("SKILL.md")).ok()?;
            let (_, meta) = ModelService::parse_skill_content(&raw);
            let description = meta.get("description").and_then(Value::as_str).map(str::to_string);
            Some((name, description))
        })
        .collect();
    out.sort();
    out
}

fn skill_names(skills_dir: &Path) -> String {
    let names: Vec<String> = list_skills(skills_dir).into_iter().map(|(n, _)| n).collect();
    if names.is_empty() { "none".to_string() } else { names.join(", ") }
}

/// Words that say what to do with a skill, not which one: "update the good
/// morning skill" names `good-morning-…`, not `update-…`.
const NOT_NAME_WORDS: &[&str] = &[
    "skill", "skills", "the", "my", "a", "an", "update", "edit", "change", "delete", "remove", "read", "show",
    "this", "that", "to", "of", "for",
];

fn name_words(name: &str) -> Vec<&str> {
    name.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty() && !NOT_NAME_WORDS.contains(w)).collect()
}

/// The existing skill `name` refers to: the exact (slugged) name, else, when
/// `loose`, the one skill sharing the most words with it ("update the good
/// morning skill" → `good-morning-workout`).  The error lists the skills
/// there are, so the model can retry.
fn find_skill(skills_dir: &Path, name: &str, loose: bool) -> anyhow::Result<String> {
    let slug = ModelService::build_skill("x", Some(name), None).name;
    if ModelService::is_system_skill(&slug) {
        anyhow::bail!("'{slug}' is a built-in skill and cannot be changed");
    }
    if skills_dir.join(&slug).join("SKILL.md").is_file() {
        return Ok(slug);
    }
    if loose {
        let wanted = name_words(&slug);
        let mut scored: Vec<(usize, String)> = list_skills(skills_dir)
            .into_iter()
            .map(|(n, _)| (name_words(&n).iter().filter(|w| wanted.contains(w)).count(), n))
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        match scored.as_slice() {
            [(_, only)] => return Ok(only.clone()),
            [(best, n), (next, _), ..] if best > next => return Ok(n.clone()),
            _ => {}
        }
    }
    anyhow::bail!("no skill named '{slug}' in this project — skills: {}", skill_names(skills_dir))
}

/// The skill an update without a `name` is about: the one whose name its new
/// content's title matches ("# Good Morning Workout" → `good-morning-workout`),
/// else the project's only skill.  Models often drop `name` when they rewrite
/// a skill they were shown; refusing sent them round the same failing call.
fn implied_skill(skills_dir: &Path, content: Option<&str>) -> anyhow::Result<String> {
    let title = content
        .map(|c| ModelService::parse_skill_content(c).0)
        .and_then(|body| body.lines().find_map(|l| l.trim().strip_prefix("# ").map(|t| t.trim().to_string())));
    if let Some(found) = title.and_then(|t| find_skill(skills_dir, &t, true).ok()) {
        return Ok(found);
    }
    match list_skills(skills_dir).as_slice() {
        [(only, _)] => Ok(only.clone()),
        _ => anyhow::bail!(
            "`name` is required — pass the name of the skill to change, one of: {}",
            skill_names(skills_dir)
        ),
    }
}

/// The skill's current text, frontmatter included.
pub fn read_skill(skills_dir: &Path, name: &str) -> anyhow::Result<String> {
    let found = find_skill(skills_dir, name, true)?;
    Ok(std::fs::read_to_string(skills_dir.join(&found).join("SKILL.md"))?)
}

/// Replace an existing skill's description and/or body, keeping its name.
pub fn update_skill(
    skills_dir: &Path,
    name: &str,
    description: Option<&str>,
    content: Option<&str>,
) -> anyhow::Result<String> {
    if description.is_none() && content.is_none() {
        anyhow::bail!("nothing to change — pass `content` and/or `description`");
    }
    let found = find_skill(skills_dir, name, true)?;
    let raw = std::fs::read_to_string(skills_dir.join(&found).join("SKILL.md"))?;
    let (old_body, meta) = ModelService::parse_skill_content(&raw);
    let old_description = meta.get("description").and_then(Value::as_str).map(str::to_string);
    // The new body's own frontmatter (if the model wrote one) must not rename it.
    let body = match content {
        Some(c) => ModelService::parse_skill_content(c).0,
        None => old_body,
    };
    let mut skill = ModelService::build_skill(&body, Some(&found), description.or(old_description.as_deref()));
    skill.name = found.clone();
    skill.id = found.clone();
    ModelService::write_skill_md_in(skills_dir, &skill)?;
    Ok(format!("Updated skill '{found}'.\n\n{}", ModelService::render_skill_content(&skill)))
}

/// Move a skill to `<skills_dir>/.deleted/<name>-<unix time>` — out of use,
/// but recoverable by hand if the request was misheard.
pub fn delete_skill(skills_dir: &Path, name: &str) -> anyhow::Result<String> {
    let found = find_skill(skills_dir, name, false)?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let trash = skills_dir.join(DELETED_DIR);
    std::fs::create_dir_all(&trash)?;
    std::fs::rename(skills_dir.join(&found), trash.join(format!("{found}-{stamp}")))?;
    Ok(format!("Deleted skill '{found}'."))
}

/// Most characters of skill text [`skills_context`] hands a step.
pub const SKILLS_CONTEXT_CHARS: usize = 12_000;

/// The current text of the project's skills, for a step about to call the
/// Skills server: an update rewrites a skill from what it says now, and a
/// read-then-write in collect cannot be relied on.  `None` when `servers`
/// holds no Skills server or the project has no skills.
pub fn skills_context(servers: &[crate::workspace::workspace::MCPServer]) -> Option<String> {
    let server = servers.iter().find(|s| s.id == SKILLS_SERVER_ID)?;
    let dir = server.config.as_ref()?.env.get(SKILLS_DIR_ENV)?;
    let dir = Path::new(dir);
    let mut text = String::from(
        "Current skills in this project (name, then its SKILL.md). This is their current text — no need to \
         list or read them. To change one, call update-skill with its `name` (as listed after ###) and the full \
         updated content; to remove one, call delete-skill with its `name`; to add one, call save-skill.\n",
    );
    let skills = list_skills(dir);
    if skills.is_empty() {
        return None;
    }
    for (name, _) in skills {
        let Ok(raw) = std::fs::read_to_string(dir.join(&name).join("SKILL.md")) else { continue };
        let entry = format!("\n### {name}\n{raw}\n");
        if text.chars().count() + entry.chars().count() > SKILLS_CONTEXT_CHARS {
            text.push_str(&format!("\n### {name}\n(text not shown — use read-skill)\n"));
            continue;
        }
        text.push_str(&entry);
    }
    Some(text)
}

fn skills_dir_from_env() -> anyhow::Result<PathBuf> {
    match std::env::var(SKILLS_DIR_ENV) {
        Ok(dir) if !dir.trim().is_empty() => Ok(PathBuf::from(dir)),
        _ => anyhow::bail!("no project skills directory is configured ({SKILLS_DIR_ENV} is unset)"),
    }
}

/// Dispatch one tool call by name.
pub fn call_tool_by_name(name: &str, args: &JsonObject) -> anyhow::Result<String> {
    match name {
        SAVE_SKILL_TOOL => {
            let skill_name = required_str(args, "name")?;
            let description = required_str(args, "description")?;
            let content = required_str(args, "content")?;
            save_skill(&skills_dir_from_env()?, skill_name, description, content)
        }
        UPDATE_SKILL_TOOL => {
            let dir = skills_dir_from_env()?;
            let content = optional_str(args, "content")?;
            let name = match optional_str(args, "name")? {
                Some(name) => name.to_string(),
                None => implied_skill(&dir, content)?,
            };
            update_skill(&dir, &name, optional_str(args, "description")?, content)
        }
        DELETE_SKILL_TOOL => delete_skill(&skills_dir_from_env()?, required_str(args, "name")?),
        READ_SKILL_TOOL => read_skill(&skills_dir_from_env()?, required_str(args, "name")?),
        LIST_SKILLS_TOOL => {
            let dir = skills_dir_from_env()?;
            let skills = list_skills(&dir);
            if skills.is_empty() {
                return Ok("No skills saved in this project yet.".to_string());
            }
            Ok(skills
                .into_iter()
                .map(|(n, d)| match d {
                    Some(d) => format!("- {n}: {d}"),
                    None => format!("- {n}"),
                })
                .collect::<Vec<_>>()
                .join("\n"))
        }
        other => anyhow::bail!("Unknown tool: {other}"),
    }
}

/// The MCP server handler exposing the skill tools.
#[derive(Debug, Clone, Default)]
pub struct SkillServer;

impl ServerHandler for SkillServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(SERVER_NAME, crate::VERSION))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(tool_definitions()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        tool_definitions().into_iter().find(|t| t.name == name)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let args = request.arguments.unwrap_or_default();
        Ok(match call_tool_by_name(&request.name, &args) {
            Ok(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
            Err(e) => CallToolResult::error(vec![ContentBlock::text(format!(
                "Error executing tool {}: {e}",
                request.name
            ))]),
        })
    }
}

/// Serve [`SkillServer`] over stdio until the client disconnects.
pub async fn run_stdio() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .try_init();
    let service = SkillServer
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await
        .map_err(|e| anyhow::anyhow!("MCP server initialization failed: {e}"))?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("slp-skill-server-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn saves_skill_md_with_frontmatter() {
        let dir = tmp_dir("save");
        let out = save_skill(
            &dir,
            "Linear Weekly Triage",
            "Triage this week's Linear issues",
            "# Linear weekly triage\n\nPull the latest issues from Linear and summarize them.",
        )
        .unwrap();
        assert!(out.contains("linear-weekly-triage"));
        let written = std::fs::read_to_string(dir.join("linear-weekly-triage/SKILL.md")).unwrap();
        assert!(written.starts_with("---"));
        assert!(written.contains("name: linear-weekly-triage"));
        assert!(written.contains("description: Triage this week's Linear issues"));
        assert!(written.contains("Pull the latest issues from Linear"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_update_without_a_name_finds_the_skill_by_its_title() {
        let dir = tmp_dir("implied");
        save_skill(&dir, "good-morning-workout", "d", "# Good Morning Workout\n\nold").unwrap();
        save_skill(&dir, "cancel-issue", "d", "# Cancel Issue\n\nx").unwrap();
        assert_eq!(implied_skill(&dir, Some("# Good Morning Workout\n\nnew")).unwrap(), "good-morning-workout");
        // No title to go on and more than one skill: say which names exist.
        let err = implied_skill(&dir, Some("just text")).unwrap_err().to_string();
        assert!(err.contains("good-morning-workout") && err.contains("cancel-issue"));
        std::fs::remove_dir_all(&dir).ok();

        let dir = tmp_dir("implied-one");
        save_skill(&dir, "good-morning-workout", "d", "# Morning\n\nold").unwrap();
        assert_eq!(implied_skill(&dir, Some("no title")).unwrap(), "good-morning-workout");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn refuses_to_overwrite_an_existing_skill() {
        let dir = tmp_dir("dup");
        save_skill(&dir, "triage", "d", "first").unwrap();
        let err = save_skill(&dir, "triage", "d", "second").unwrap_err();
        assert!(err.to_string().contains("already exists"));
        let written = std::fs::read_to_string(dir.join("triage/SKILL.md")).unwrap();
        assert!(written.contains("first"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn refuses_reserved_names() {
        let dir = tmp_dir("reserved");
        assert!(save_skill(&dir, "__system_mcp__", "d", "x").is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn requires_all_arguments() {
        let mut args = JsonObject::new();
        args.insert("name".into(), json!("x"));
        assert!(call_tool_by_name(SAVE_SKILL_TOOL, &args).is_err());
    }

    #[test]
    fn update_keeps_name_and_untouched_fields() {
        let dir = tmp_dir("update");
        save_skill(&dir, "good-morning-workout", "Use when the user says good morning", "# Morning\n\n10 min").unwrap();
        // "morning" finds the one skill whose name contains it.
        update_skill(&dir, "morning", None, Some("---\nname: other\n---\n# Morning\n\n10 min, then a cool-down"))
            .unwrap();
        let written = std::fs::read_to_string(dir.join("good-morning-workout/SKILL.md")).unwrap();
        assert!(written.contains("name: good-morning-workout"));
        assert!(written.contains("description: Use when the user says good morning"));
        assert!(written.contains("cool-down"));
        assert!(!dir.join("other").exists());
        assert!(update_skill(&dir, "good-morning-workout", None, None).is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_request_phrased_name_finds_the_skill_by_shared_words() {
        let dir = tmp_dir("words");
        save_skill(&dir, "good-morning-workout", "d", "x").unwrap();
        save_skill(&dir, "linear-weekly-triage", "d", "y").unwrap();
        assert!(read_skill(&dir, "update_good_morning_skill").unwrap().contains("good-morning-workout"));
        assert!(read_skill(&dir, "the linear skill").unwrap().contains("linear-weekly-triage"));
        assert!(read_skill(&dir, "update skill").is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn skills_context_reads_the_dir_from_the_server_config() {
        use crate::workspace::workspace::{MCPServer, MCPServerConfig};
        let dir = tmp_dir("context");
        save_skill(&dir, "good-morning-workout", "d", "# Morning\n\nten minutes").unwrap();
        let mut config = MCPServerConfig::default();
        config.env.insert(SKILLS_DIR_ENV.to_string(), dir.to_string_lossy().into_owned());
        let server = MCPServer { id: SKILLS_SERVER_ID.to_string(), config: Some(config), ..Default::default() };
        let text = skills_context(&[server]).unwrap();
        assert!(text.contains("### good-morning-workout") && text.contains("ten minutes"));
        assert!(skills_context(&[MCPServer::default()]).is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn unknown_names_list_the_skills_there_are() {
        let dir = tmp_dir("unknown");
        save_skill(&dir, "triage", "d", "x").unwrap();
        let err = read_skill(&dir, "nothing-like-it").unwrap_err().to_string();
        assert!(err.contains("skills: triage"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn delete_needs_the_exact_name_and_keeps_a_copy() {
        let dir = tmp_dir("delete");
        save_skill(&dir, "good-morning-workout", "d", "x").unwrap();
        assert!(delete_skill(&dir, "morning").is_err());
        delete_skill(&dir, "good-morning-workout").unwrap();
        assert!(!dir.join("good-morning-workout").exists());
        assert!(list_skills(&dir).is_empty());
        let kept: Vec<_> = std::fs::read_dir(dir.join(DELETED_DIR)).unwrap().collect();
        assert_eq!(kept.len(), 1);
        assert!(delete_skill(&dir, "bash").is_err());
        std::fs::remove_dir_all(dir).ok();
    }
}
