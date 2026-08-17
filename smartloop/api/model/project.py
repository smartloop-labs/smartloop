"""Project API schemas"""

from typing import List, Optional
from pydantic import BaseModel, Field

from smartloop.api.model.mcp import MCPServerResponse
from smartloop.api.model.chat import DocumentResponse


class SkillRequest(BaseModel):
    """Request model for creating a skill.  Only *content* is required;
    name and description are auto-generated from content when omitted."""
    content: str = Field(..., description="Skill content")
    name: Optional[str] = Field(default=None, description="Skill name (auto-generated if omitted)")
    description: Optional[str] = Field(default=None, description="Skill description (auto-generated if omitted)")

class UpdateSkillRequest(BaseModel):
    """Request model for updating an existing skill.
    Content should include YAML frontmatter with name and description."""
    content: str = Field(..., description="Full skill content with YAML frontmatter")

class SkillResponse(BaseModel):
    """Response model for project skills"""
    id: str = Field(..., description="Skill ID")
    name: str = Field(..., description="Skill name")
    description: Optional[str] = Field(default=None, description="Skill description")
    content: str = Field(..., description="Skill content")

class CreateProjectRequest(BaseModel):
    """Request to create a new project"""
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(default=None, description="Project description")
    developer_token: Optional[str] = Field(default=None, description="Developer token for model download")
    system: bool = Field(default=False, description="Whether this is a system project")
    skills: List[SkillRequest] = Field(default=[], description="Initial list of skills for the project")
    cwd: Optional[str] = Field(default=None, description="Working directory for the project's file tools; defaults to ~/smartloop/<name> when omitted")


class UpdateProjectRequest(BaseModel):
    """Request to update an existing project"""
    name: Optional[str] = Field(default=None, description="New project name")
    description: Optional[str] = Field(default=None, description="New project description")
    skills: List[SkillRequest] = Field(default=[], description="Updated list of skills for the project")
    web_search_enabled: Optional[bool] = Field(default=None, description="Enable/disable web search for this project; omit to leave unchanged")

class ProjectResponse(BaseModel):
    """Single project info"""
    id: str
    name: Optional[str] = None
    cwd: Optional[str] = Field(default=None, description="Visible project directory path")
    description: Optional[str] = None
    model_name: Optional[str] = None
    skills: List[SkillResponse] = []
    mcp: List[MCPServerResponse] = []
    documents: List[DocumentResponse] = []
    current: bool = False
    system: bool = False
    # Mirrors Project.web_search_enabled — on unless explicitly disabled.
    web_search_enabled: bool = True


class ProjectListResponse(BaseModel):
    """List of projects"""
    projects: List[ProjectResponse]


class DeleteProjectResponse(BaseModel):
    """Response after deleting a project"""
    id: str
    deleted: bool
    message: str
