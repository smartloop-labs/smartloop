//! Port of `smartloop/workspace/__init__.py`.

pub mod context;
pub mod migrations;
pub mod types;
pub mod workspace;

pub use context::{
    ProjectBoundExecutor, ProjectContext, active_project_id, bind, owns_workspace, pin_project,
    pinned_project_id, reset_active_project, set_active_project, use_project,
};
pub use types::{Document, DocumentInfo, DocumentStatus, Skill};
pub use workspace::{
    MCPAuthType, MCPOAuthCredentials, MCPServer, MCPServerConfig, MCPServerStatus, MCPServerType,
    MCPToolDefinition, Project, RemoteMCPServerConfig, Workspace, WorkspaceModel,
};
