"""API Routers"""

from .auth import router as auth_router
from .chat import router as chat_router
from .documents import router as documents_router
from .projects import router as projects_router
from .assets import router as assets_router
from .mcp_servers import router as mcp_servers_router
from .models import router as models_router
from .init import router as init_router
from .skills import router as skills_router
from .users import router as users_router
from .feedback import router as feedback_router
from .project_io import router as project_io_router
from .releases import router as releases_router

__all__ = ["auth_router", "chat_router", "documents_router", "projects_router", "assets_router", "mcp_servers_router", "models_router", "init_router", "skills_router", "users_router", "feedback_router", "project_io_router", "releases_router"]
