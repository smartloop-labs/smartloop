#!/usr/bin/env python3
"""
Smartloop - An SLM framework for inferencing and tuning models on edge devices.

The agent is started automatically if not running, and stays running
to avoid reloading the model on each invocation.
"""

import argparse
import logging
import os
import certifi
import multiprocessing

from version import __version__
from smartloop.auth import credential_store

from commands.base import Command
from commands.console import settings
from commands.model import ModelCommand
from commands.run import RunCommand
from commands.token import TokenCommand
from commands.agent import AgentCommand


# Use certifi CA bundle for SSL verification (required for PyInstaller builds
# where system certificates are not available)
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("smartloop").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Command Handler
# ---------------------------------------------------------------------------

class CommandHandler(
    ModelCommand,
    RunCommand,
    TokenCommand,
    AgentCommand,
):
    """Dispatches CLI commands via HTTP to the running API server."""

    def __init__(
        self,
        args,
        host: str,
        port: int,
        model_name: str,
        project_id: str | None,
        project_name: str | None,
        developer_token: str,
        project_dir: str,
        parser,
        agent_parser,
    ):
        self.args = args
        self.host = host
        self.port = port
        self.model_name = model_name
        self.project_id = project_id
        self.project_name = project_name
        self.developer_token = developer_token
        self.project_dir = project_dir
        self.parser = parser
        self.agent_parser = agent_parser

    def dispatch(self) -> None:
        """Resolve and invoke the correct command handler."""

        command_maps = {
            "run": RunCommand,
            "status": ModelCommand,
            "token": TokenCommand,
            "agent": AgentCommand,
        }

        if not self.args.command or getattr(self.args, "resume", None):
            RunCommand.execute(self)
        else:
            command_maps.get(self.args.command).execute(self)

def main():
    """Main Command Line entry point."""
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="AI Agent harness for extracting information and creating new content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--resume", metavar="CONVERSATION_ID", default=None,
                        help="Resume a previous conversation by session ID")
    parser.add_argument("-C", "--path", default=None, metavar="PATH",
                        help="Working directory to use (defaults to current directory)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
 
    # Run command
    run_parser = subparsers.add_parser("run", help="Start interactive chat session")
    run_parser.add_argument("--model", "-m", help="Model name (default: $SLP_BASE_MODEL_NAME or gemma3-1b)")
    run_parser.add_argument("--project-name", help="Project name (default: $SLP_PROJECT_NAME)")
    run_parser.add_argument("--host", help="API server host (default: $API_HOST or 127.0.0.1)")
    run_parser.add_argument("--port", "-p", type=int, help="API server port (default: $SLP_API_PORT or 38540)")
    run_parser.add_argument("--resume", metavar="CONVERSATION_ID", default=None,
                           help="Resume a previous conversation by session ID")

    # Status command
    subparsers.add_parser("status", help="Show project status")

    # Agent management commands
    agent_parser = subparsers.add_parser("agent", help="Agent management commands (start/stop/status/restart)")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", help="Agent commands")
    start_agent_parser = agent_subparsers.add_parser("start", help="Start the agent in background")
    start_agent_parser.add_argument("--no-service", action="store_true", help="Disable auto-restart on crash")
    start_agent_parser.add_argument("--host", help="API server host (default: $SLP_API_HOST or 127.0.0.1)")
    start_agent_parser.add_argument("--port", "-p", type=int, help="API server port (default: $SLP_API_PORT or 38540)")
    agent_subparsers.add_parser("stop", help="Stop the background agent")
    agent_subparsers.add_parser("status", help="Show agent status")
    restart_agent_parser = agent_subparsers.add_parser("restart", help="Restart the agent")
    restart_agent_parser.add_argument("--no-service", action="store_true", help="Disable auto-restart on crash")
    restart_agent_parser.add_argument("--host", help="API server host (default: $SLP_API_HOST or 127.0.0.1)")
    restart_agent_parser.add_argument("--port", "-p", type=int, help="API server port (default: $SLP_API_PORT or 38540)")

    # Token management commands
    token_parser = subparsers.add_parser("token", help="Manage your developer token")
    token_subparsers = token_parser.add_subparsers(dest="token_command", help="Token commands")
    token_set_parser = token_subparsers.add_parser("set", help="Set your developer token")
    token_set_parser.add_argument("token_value", nargs="?", default=None, help="Developer token (prompted if omitted)")
    token_subparsers.add_parser("clear", help="Remove stored token")

    args = parser.parse_args()

    # Persist -t token to keychain so it's available to all consumers
    cli_token = getattr(args, "developer_token", None)
    if cli_token:
        credential_store.set_token(cli_token)

    # Re-resolve after potential keychain write so the value reflects -t
    developer_token = cli_token or os.environ.get("SLP_DEVELOPER_TOKEN")

    # Resolve configuration
    model_name = getattr(args, "model", None) or os.environ.get("SLP_BASE_MODEL_NAME")
    project_id = os.environ.get("SLP_PROJECT_ID")
    project_name = getattr(args, "project_name", None) or os.environ.get("SLP_PROJECT_NAME")
    host = getattr(args, "host", None) or os.environ.get("SLP_API_HOST", "127.0.0.1")
    port = getattr(args, "port", None) or settings.api_port

    # Project working directory: explicit -C overrides, otherwise ~/smartloop/<slug>
    if args.path:
        target = os.path.abspath(args.path)
        if not os.path.isdir(target):
            parser.error(f"path '{args.path}' is not a valid directory")
        project_dir = target
    else:
        base = os.path.expanduser("~/smartloop")
        if project_name:
            slug = project_name.lower().replace(" ", "_")
            project_dir = os.path.join(base, slug)
        else:
            project_dir = base

    CommandHandler(
        args=args,
        host=host,
        port=port,
        model_name=model_name,
        project_id=project_id,
        project_name=project_name,
        developer_token=developer_token,
        project_dir=project_dir,
        parser=parser,
        agent_parser=agent_parser,
    ).dispatch()

if __name__ == "__main__":
    main()
