"""
Model configuration classes for the core framework.
"""
from typing import Optional, Annotated

from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Configuration for model initialization"""
    model_name: Annotated[str, Field(description="model name")]
    project_id: Annotated[Optional[str], Field(default='d4c8f5a2-7b3e-4f91-a6d2-8e9c1b4a7f3d', description="Project id")]
    quantize: Annotated[str, Field(default='q4_k_m', description="Quantization level")]
    hf_token: Annotated[Optional[str], Field(default=None, description="HuggingFace token")]

    # Marked exclude = True to avoid it being serialized to JSON, persists in memory only
    developer_token: Annotated[Optional[str], Field(default=None, exclude=True, description="Developer token for authentication")]