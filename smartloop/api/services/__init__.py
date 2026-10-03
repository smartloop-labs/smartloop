"""API Services"""

from typing import Optional
from .model_service import ModelService
from .api_client import SLPAPIClient
from .storage_service import compute_usage, discard_partial_download, run_cleanup

__all__ = [
    "ModelService", "get_model_service", "SLPAPIClient", "get_api_client",
    "compute_usage", "run_cleanup", "discard_partial_download",
]


# Global singleton instances
_model_service: Optional[ModelService] = None
_api_client: Optional[SLPAPIClient] = None

def get_model_service() -> ModelService:
    """Get the global ModelService singleton"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service

def get_api_client() -> SLPAPIClient:
    """Get the global SLPAPIClient singleton"""
    global _api_client
    if _api_client is None:
        _api_client = SLPAPIClient()
    return _api_client