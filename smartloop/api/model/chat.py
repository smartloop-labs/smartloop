"""OpenAI-compatible chat completion schemas"""

import time
import uuid
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation"""
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message author"
    )
    content: Optional[str] = Field(default=None, description="The content of the message")
    attachments: Optional[List[str]] = Field(
        default=None,
        description="List of asset IDs to include as context in this message",
    )


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(
        ..., description="A list of messages comprising the conversation"
    )
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation tracking")
    message_id: Optional[str] = Field(default=None, description="Client-provided message ID for the assistant response")
    project_id: Optional[str] = Field(default=None, description="Project ID to use for this request")
    cwd: Optional[str] = Field(default=None, description="Current working directory for file tools")
    stream: bool = Field(default=False, description="Whether to stream the response")
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, description="Maximum tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: Optional[float] = Field(
        default=0.0, ge=-2.0, le=2.0, description="Presence penalty"
    )
    stop: Optional[List[str]] = Field(
        default=None, description="Stop sequences"
    )


class DeltaContent(BaseModel):
    """Delta content for streaming responses"""
    role: Optional[str] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    """A single choice in a streaming chunk response"""
    index: int = 0
    delta: DeltaContent
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk response"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChunkChoice]


class Choice(BaseModel):
    """A single choice in a non-streaming response"""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Citation(BaseModel):
    """Reference to a source document used in the response."""
    document_id: str
    document_name: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible non-streaming response"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    message_id: Optional[str] = None
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)
    citations: Optional[List[Citation]] = None


# --- Document models ---

class AddDocumentRequest(BaseModel):
    """Request to add a document source"""
    source: str = Field(..., description="File path or URL for the document source")
    cwd: Optional[str] = Field(default=None, description="Current working directory")
    generate_filename: bool = Field(default=False, description="Use LLM to generate a descriptive filename for txt files from the document content")


class DocumentResponse(BaseModel):
    """Single document info"""
    id: Optional[str] = None
    path: Optional[str] = None
    status: str = "pending"
    summary: Optional[str] = None
    filename: Optional[str] = None


class DocumentListResponse(BaseModel):
    """List of documents"""
    documents: List[DocumentResponse]


class DeleteDocumentResponse(BaseModel):
    """Response after deleting a document"""
    id: str
    deleted: bool
    message: str


# --- Asset models ---

class AssetUploadResponse(BaseModel):
    """Response after uploading an asset"""
    asset_id: str
    filename: str
    content_type: str
    size: int = Field(..., description="Original file size in bytes")
    markdown: bool = Field(
        default=False,
        description="True when the file was converted to markdown",
    )


class AssetResponse(BaseModel):
    """Metadata and content for a stored asset"""
    asset_id: str
    filename: str
    content_type: str
    size: int
    markdown: bool
    content: Optional[str] = Field(default=None, description="Parsed markdown content")


class DeleteAssetResponse(BaseModel):
    """Response after deleting an asset"""
    asset_id: str
    deleted: bool
    message: str


# --- Feedback models ---

class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session ID of the conversation")
    message_id: str = Field(..., description="Message ID of the assistant response")
    project_id: Optional[str] = Field(default=None, description="Project ID to scope the feedback to")
    rating: Literal["up", "down"] = Field(..., description="Thumbs up or down")


class FeedbackResponse(BaseModel):
    feedback_id: str
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    indexed: bool = Field(default=False, description="Whether the Q&A pair was added to or removed from the vector index")


