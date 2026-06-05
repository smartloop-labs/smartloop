"""CLI command classes."""

from version import __version__

from .base import Command
from .document import DocumentCommand
from .skill import SkillCommand
from .model import ModelCommand
from .run import RunCommand
from .token import TokenCommand
from .mcp import McpCommand
from .agent import AgentCommand
from .projects import ProjectsCommand
from .console import settings

settings.version = __version__

__all__ = [
    "Command",
    "DocumentCommand",
    "SkillCommand",
    "ModelCommand",
    "RunCommand",
    "TokenCommand",
    "McpCommand",
    "AgentCommand",
    "ProjectsCommand",
]
