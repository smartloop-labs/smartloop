//! Port of `smartloop/api/model/project.py`.
//!
//! Project API schemas.

use serde::{Deserialize, Serialize};

use crate::api::model::chat::DocumentResponse;
use crate::api::model::mcp::MCPServerResponse;

/// Request model for creating a skill.  Only `content` is required;
/// name and description are auto-generated from content when omitted.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SkillRequest {
    /// Skill content.
    pub content: String,
    /// Skill name (auto-generated if omitted).
    #[serde(default)]
    pub name: Option<String>,
    /// Skill description (auto-generated if omitted).
    #[serde(default)]
    pub description: Option<String>,
}

/// Request model for updating an existing skill.
/// Content should include YAML frontmatter with name and description.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct UpdateSkillRequest {
    /// Full skill content with YAML frontmatter.
    pub content: String,
}

/// Response model for project skills.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SkillResponse {
    /// Skill ID.
    pub id: String,
    /// Skill name.
    pub name: String,
    /// Skill description.
    #[serde(default)]
    pub description: Option<String>,
    /// Skill content.
    pub content: String,
}

/// Request to create a new project.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct CreateProjectRequest {
    /// Project name.
    pub name: String,
    /// Project description.
    #[serde(default)]
    pub description: Option<String>,
    /// Developer token for model download.
    #[serde(default)]
    pub developer_token: Option<String>,
    /// Whether this is a system project.
    #[serde(default)]
    pub system: bool,
    /// Initial list of skills for the project.
    #[serde(default)]
    pub skills: Vec<SkillRequest>,
    /// Working directory for the project's file tools; defaults to
    /// `~/smartloop/<name>` when omitted.
    #[serde(default)]
    pub cwd: Option<String>,
}

/// Request to update an existing project.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct UpdateProjectRequest {
    /// New project name.
    #[serde(default)]
    pub name: Option<String>,
    /// New project description.
    #[serde(default)]
    pub description: Option<String>,
    /// Updated list of skills for the project.
    #[serde(default)]
    pub skills: Vec<SkillRequest>,
    /// Enable/disable web search for this project; omit to leave unchanged.
    #[serde(default)]
    pub web_search_enabled: Option<bool>,
}

fn default_true() -> bool {
    true
}

/// Single project info.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectResponse {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// Visible project directory path.
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub skills: Vec<SkillResponse>,
    #[serde(default)]
    pub mcp: Vec<MCPServerResponse>,
    #[serde(default)]
    pub documents: Vec<DocumentResponse>,
    #[serde(default)]
    pub current: bool,
    #[serde(default)]
    pub system: bool,
    /// Mirrors `Project.web_search_enabled` — on unless explicitly disabled.
    #[serde(default = "default_true")]
    pub web_search_enabled: bool,
}

impl Default for ProjectResponse {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: None,
            cwd: None,
            description: None,
            model_name: None,
            skills: Vec::new(),
            mcp: Vec::new(),
            documents: Vec::new(),
            current: false,
            system: false,
            web_search_enabled: true,
        }
    }
}

/// List of projects.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectListResponse {
    pub projects: Vec<ProjectResponse>,
}

/// Response after deleting a project.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DeleteProjectResponse {
    pub id: String,
    pub deleted: bool,
    pub message: String,
}
