"""AgentCommand — ``agent`` CLI sub-command."""

from __future__ import annotations

import time

from smartloop.constants import SLP_PRIMARY
from smartloop.server import is_server_running, stop_server, start_server, get_status

from commands.base import Command
from commands.console import console


class AgentCommand(Command):
    """Handles ``agent`` CLI sub-commands (start / stop / status / restart)."""

    args: object
    host: str
    port: int
    agent_parser: object

    def execute(self) -> None:
        """Dispatch agent sub-commands."""
        sub = getattr(self, f"agent_{self.args.agent_command}", None)
        if sub:
            sub()
        else:
            self.agent_parser.print_help()

    def agent_start(self) -> None:
        if is_server_running(self.host, self.port):
            console.print(f"[{SLP_PRIMARY}]Agent already running at http://{self.host}:{self.port}[/{SLP_PRIMARY}]")
        else:
            start_server(
                self.host, self.port,
                service=not getattr(self.args, "no_service", False),
            )

    def agent_stop(self) -> None:
        stop_server()

    def agent_status(self) -> None:
        status = get_status(self.host, self.port)
        if status["running"]:
            console.print(f"[{SLP_PRIMARY}]Agent running at http://{self.host}:{self.port}[/{SLP_PRIMARY}]")
            console.print(f"  PID: {status.get('pid') or 'unknown'}")
            console.print(f"  Model loaded: {status.get('model_loaded', False)}")
            console.print(f"  Model name: {status.get('model_name', 'none')}")
        else:
            console.print("[dim]Agent not running[/dim]")

    def agent_restart(self) -> None:
        stop_server()
        time.sleep(1)
        start_server(
            self.host, self.port,
            service=not getattr(self.args, "no_service", False),
        )
