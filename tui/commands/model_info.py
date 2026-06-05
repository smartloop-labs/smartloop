"""ModelInfoMixin — /model list|switch commands."""

from __future__ import annotations

import asyncio

import httpx
from rich.table import Table
from textual import work
from textual.containers import VerticalScroll
from textual.widgets import Static

from smartloop.auth import credential_store
from tui.events import (
    BootstrapProgress,
    BootstrapStatus,
    BootstrapComplete,
    BootstrapError,
    parse_bootstrap_sse,
)


def _format_bytes(n: int) -> str:
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    return f"{n / 1024:.0f} KB"


class ModelInfo:
    """Command handler for /model (list|switch)."""

    server_url: str
    model_name: str

    def _handle_model_command(self, args: str) -> None:
        """Dispatch /model sub-commands."""
        args = args.strip()
        if not args or args == "list":
            self._model_list()
        elif args.startswith("switch "):
            index_str = args[7:].strip()
            if index_str:
                self._model_switch(index_str)
            else:
                self._append_system("Usage: /model switch <#>")
        else:
            self._append_system("Usage: /model [list|switch <#>]")

    @work(exclusive=True)
    async def _model_list(self) -> None:
        """List available models from the platform."""
        self._set_loading("Fetching models...")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{self.server_url}/v1/models")
                resp.raise_for_status()
                models = resp.json()

                current = ""
                try:
                    health = await client.get(f"{self.server_url}/health")
                    if health.is_success:
                        current = health.json().get("model_name") or ""
                except httpx.RequestError:
                    pass

            if not models:
                self._append_system("No models available")
                return

            table = Table(style="#8b949e")
            table.add_column("#", style="dim", width=3)
            table.add_column("Name")
            table.add_column("Description")
            table.add_column("Access")
            table.add_column("Current")
            for i, m in enumerate(models, 1):
                access = m.get("access_level") or ""
                if not m.get("is_accessible", False):
                    access = f"{access} (locked)".strip()
                table.add_row(
                    str(i),
                    m.get("name", ""),
                    (m.get("description") or "").strip(),
                    access,
                    "yes" if m.get("name") == current else "",
                )
            log = self.query_one("#chat-log", VerticalScroll)
            log.mount(Static(table, classes="system-msg"))
            log.scroll_end(animate=False)
        except (httpx.RequestError, httpx.HTTPStatusError):
            self._append_system("Request failed")
        finally:
            self._clear_loading()

    @work(exclusive=True)
    async def _model_switch(self, index_str: str) -> None:
        """Switch to a model by its index — init/download it, tracking SSE progress."""
        try:
            index = int(index_str)
        except ValueError:
            self._append_system("Invalid index. Use /model list to see numbered models.")
            return

        log = self.query_one("#chat-log", VerticalScroll)
        download_label: Static | None = None
        progress_widget: Static | None = None

        self._set_loading("Fetching models...")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{self.server_url}/v1/models")
                resp.raise_for_status()
                models = resp.json()

            if index < 1 or index > len(models):
                self._append_system(f"Invalid index {index}. Models have {len(models)} entries.")
                return

            model = models[index - 1]
            target = model.get("name", "")

            payload: dict = {"model_name": target}
            token = credential_store.get_token()
            if token:
                payload["developer_token"] = token

            self._update_loading(f"Switching to {target}...")

            async with httpx.AsyncClient(timeout=1800) as client:
                async with client.stream(
                    "POST",
                    f"{self.server_url}/v1/init",
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        detail = ""
                        try:
                            detail = response.json().get("detail", "")
                        except Exception:
                            pass
                        self._append_system(f"Failed to switch model{': ' + detail if detail else ''}")
                        return

                    async for event in parse_bootstrap_sse(response):
                        match event:
                            case BootstrapProgress(filename=fn, downloaded=dl, total=total):
                                if download_label is None:
                                    download_label = Static("", classes="bootstrap-status")
                                    await log.mount(download_label)
                                if progress_widget is None:
                                    progress_widget = Static("", classes="bootstrap-progress")
                                    await log.mount(progress_widget)
                                pct = (dl / total * 100) if total else 0
                                bar = self._render_block_bar(dl, total)
                                progress_widget.update(f"{bar} [#8b949e]{pct:3.0f}%[/#8b949e]")
                                download_label.update(
                                    f"Downloading {fn}  {_format_bytes(dl)} / {_format_bytes(total)}"
                                )
                                log.scroll_end(animate=False)

                            case BootstrapStatus(status=st, message=msg):
                                _done = (
                                    "download_complete", "model_ready",
                                    "creating_project", "loading", "model_loaded",
                                )
                                if st in _done:
                                    if progress_widget is not None:
                                        await progress_widget.remove()
                                        progress_widget = None
                                    if download_label is not None:
                                        await download_label.remove()
                                        download_label = None
                                self._update_loading(msg or "Working...")

                            case BootstrapComplete(model_name=mn, project=proj):
                                if mn:
                                    self.model_name = mn
                                    self.sub_title = mn
                                if proj:
                                    self.project_id = proj.get("id")
                                    pname = proj.get("name", "")
                                    self.project_rules = proj.get("rules", "")
                                    if pname:
                                        self.title = pname

                            case BootstrapError(message=msg):
                                self._append_system(f"Failed to switch model: {msg}")
                                return

            # Wait until the server reports the target model actually loaded
            # before declaring success, so the first chat after a switch does
            # not race the model load.
            self._update_loading(f"Loading {target}...")
            loaded_name = ""
            model_ready = False
            for _ in range(60):  # up to ~30s
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        health = await client.get(f"{self.server_url}/health")
                    if health.is_success:
                        h = health.json()
                        loaded_name = h.get("model_name") or loaded_name
                        if h.get("model_loaded") and loaded_name == target:
                            model_ready = True
                            break
                except httpx.RequestError:
                    pass
                await asyncio.sleep(0.5)

            if loaded_name:
                self.model_name = loaded_name
                self.sub_title = loaded_name
            self._refresh_info_bar()
            if model_ready:
                self._append_system(f"Switched to model: {self.model_name}")
            else:
                self._append_system(
                    f"Model {target} initialized but not ready yet — "
                    "give it a moment before chatting."
                )
        except (httpx.RequestError, httpx.HTTPStatusError):
            self._append_system("Request failed")
        finally:
            if progress_widget is not None:
                await progress_widget.remove()
            if download_label is not None:
                await download_label.remove()
            self._clear_loading()
