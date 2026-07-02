"""Auth mixin — /token show|set|clear commands."""

from __future__ import annotations

import httpx
from textual import work

class Auth:
    """Command handler for token/auth related slash commands."""

    server_url: str

    def _handle_token_command(self, args: str) -> None:
        """Dispatch /token sub-commands."""
        if args == "set" or args.startswith("set "):
            value = args[4:].strip() if args.startswith("set ") else ""
            self._token_set(value)
        elif args == "clear":
            self._token_clear()
        elif args in ("", "show"):
            self._token_show()
        else:
            self._append_system("Usage: /token [show|set|clear]")

    @work(exclusive=True)
    async def _token_show(self) -> None:
        """Display the current auth status via the local server."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{self.server_url}/v1/auth/status")
                resp.raise_for_status()
                data = resp.json()
        except (httpx.RequestError, httpx.HTTPStatusError):
            self._append_system("Failed to read auth status")
            return

        if data.get("authenticated"):
            method = data.get("method") or "unknown"
            user = data.get("user") or {}
            who = user.get("email") or user.get("name")
            line = f"[green]Authenticated[/green] via {method}"
            if who:
                line += f" as {who}"
            self._append_system(line)
        else:
            self._append_system(
                "[dim]No token set. Use [bold]/token set[/bold] to add one.\n"
                "Get your token at[/dim] [bold]https://app.smartloop.ai/developer[/bold]"
            )

    @work(exclusive=True)
    async def _token_set(self, value: str) -> None:
        """Set the developer token via the local server."""
        if not value:
            self._append_system(
                "Usage: /token set <token>\n"
                "[dim]Get your token at[/dim] [bold]https://app.smartloop.ai/developer[/bold]"
            )
            return
        self._set_loading("Saving token...")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{self.server_url}/v1/auth/token", json={"token": value}
                )
                resp.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError):
            self._append_system("Failed to save token")
            return
        finally:
            self._clear_loading()
        masked = value[:8] + "..." + value[-3:] if len(value) > 14 else "***"
        self._append_system(f"[green]Token saved:[/green] {masked}")

    @work(exclusive=True)
    async def _token_clear(self) -> None:
        """Remove all stored tokens (developer, access, refresh) via the server."""
        self._set_loading("Clearing tokens...")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.delete(f"{self.server_url}/v1/auth/token")
                resp.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError):
            self._append_system("Failed to clear tokens")
            return
        finally:
            self._clear_loading()
        self._append_system("[green]All tokens cleared.[/green]")
