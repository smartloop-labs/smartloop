//! Port of `smartloop/api/model/__init__.py`.
//!
//! Request/response models for the HTTP API.

pub mod chat;
pub mod mcp;
pub mod project;

pub use chat::{
    AddDocumentRequest, AdditionalMessageParams, AssetResponse, AssetUploadResponse,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    ChatStatus, ChatStatusChunk, Choice, ChunkChoice, Citation, CitationSource,
    DeleteAssetResponse, DeleteDocumentResponse, DeltaContent, DocumentListResponse,
    DocumentResponse, FeedbackRequest, FeedbackResponse, Rating, Role, Usage,
};
pub use mcp::{
    DeleteMCPServerResponse, MCPDefinitionRequest, MCPOAuthCallbackRequest,
    MCPRegisterRequest, MCPRegisterResponse, MCPServerListResponse, MCPServerResponse,
    MCPToolCallRequest, MCPToolCallResponse, MCPToolResponse,
};
pub use project::{
    CreateProjectRequest, DeleteProjectResponse, ProjectListResponse, ProjectResponse,
    SkillRequest, SkillResponse, UpdateProjectRequest, UpdateSkillRequest,
};
