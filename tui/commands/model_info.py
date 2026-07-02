"""ModelInfoMixin — /model list|switch commands."""

from __future__ import annotations

import httpx
from rich.table import Table
from textual import work
from textual.containers import VerticalScroll
from textual.widgets import Static

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
        elif args.startswith("enable "):
            index_str = args[7:].strip()
            if index_str:
                self._model_set_enabled(index_str, True)
            else:
                self._append_system("Usage: /model enable <#>")
        elif args.startswith("disable "):
            index_str = args[8:].strip()
            if index_str:
                self._model_set_enabled(index_str, False)
            else:
                self._append_system("Usage: /model disable <#>")
        else:
            self._append_system("Usage: /model [list|enable <#>|disable <#>]")

    @work(exclusive=True)
    async def _model_list(self) -> None:
        """List available models from the platform."""
        self._set_loading("Fetching models...")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{self.server_url}/v1/models")
                resp.raise_for_status()
                models = resp.json()

                # Cross-reference workspace state: a model only counts as
                # enabled when it's enabled *and* downloaded — the orchestrator
                # can't use weights that aren't on disk.
                ws_lookup: dict = {}
                try:
                    ws_resp = await client.get(f"{self.server_url}/v1/models/workspace")
                    if ws_resp.is_success:
                        ws_lookup = {w.get("model_name"): w for w in ws_resp.json()}
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
            table.add_column("Enabled")
            for i, m in enumerate(models, 1):
                access = m.get("access_level") or ""
                if not m.get("is_accessible", False):
                    access = f"{access} (locked)".strip()
                ws = ws_lookup.get(m.get("name")) or {}
                is_enabled = bool(ws.get("enabled") and ws.get("downloaded"))
                table.add_row(
                    str(i),
                    m.get("name", ""),
                    (m.get("description") or "").strip(),
                    access,
                    "yes" if is_enabled else "no",
                )
            log = self.query_one("#chat-log", VerticalScroll)
            log.mount(Static(table, classes="system-msg"))
            log.scroll_end(animate=False)
        except (httpx.RequestError, httpx.HTTPStatusError):
            self._append_system("Request failed")
        finally:
            self._clear_loading()

    @work(exclusive=True)
    async def _model_set_enabled(self, index_str: str, enabled: bool) -> None:
        """Enable or disable a model by its index.

        Enabled models are the ones the orchestrator is allowed to pick from
        when running a task. Enabling a model that isn't on disk yet downloads
        it first (tracking SSE progress); disabling just flips the flag.
        """
        action = "enable" if enabled else "disable"
        try:
            index = int(index_str)
        except ValueError:
            self._append_system(f"Invalid index. Use /model list to see numbered models.")
            return

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

            if enabled and not model.get("is_accessible", True):
                self._append_system(f"{target} is not available on your plan.")
                return

            # Enabling requires the model present locally — download if needed.
            if enabled and not await self._ensure_model_downloaded(target):
                return

            self._update_loading(f"{'Enabling' if enabled else 'Disabling'} {target}...")
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.patch(
                    f"{self.server_url}/v1/models/workspace/{target}",
                    json={"enabled": enabled},
                )
            if resp.is_success:
                self._append_system(f"{'Enabled' if enabled else 'Disabled'} model: {target}")
            else:
                detail = ""
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    pass
                self._append_system(f"Failed to {action} {target}{': ' + detail if detail else ''}")
        except (httpx.RequestError, httpx.HTTPStatusError):
            self._append_system("Request failed")
        finally:
            self._clear_loading()

    async def _ensure_model_downloaded(self, target: str) -> bool:
        """Download a model (no-op if already on disk) via the init SSE stream.

        Returns True on success. The orchestrator base model stays loaded; this
        only fetches/registers the target so it can be enabled.
        """
        log = self.query_one("#chat-log", VerticalScroll)
        download_label: Static | None = None
        progress_widget: Static | None = None

        payload: dict = {"model_name": target}

        self._update_loading(f"Preparing {target}...")
        try:
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
                        self._append_system(f"Failed to download {target}{': ' + detail if detail else ''}")
                        return False

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
                                    f"Downloading {self._friendly_filename(fn)}  {_format_bytes(dl)} / {_format_bytes(total)}"
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

                            case BootstrapError(message=msg):
                                self._append_system(f"Failed to download {target}: {msg}")
                                return False
            return True
        finally:
            if progress_widget is not None:
                await progress_widget.remove()
            if download_label is not None:
                await download_label.remove()
