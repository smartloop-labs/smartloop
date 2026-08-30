import os
import threading
from typing import Optional

from smartloop.platform.client import ModelAccessError, SLPPlatformClient
from smartloop.platform.service import PlatformService
from smartloop.platform.workspace_db import WorkspaceDB

__all__ = ["ModelAccessError", "SLPPlatformClient", "PlatformService", "WorkspaceDB",
           "get_platform_service"]


_platform_service: Optional[PlatformService] = None
_platform_service_db_path: Optional[str] = None
_platform_service_lock = threading.Lock()


def get_platform_service() -> PlatformService:
    """Return the global PlatformService singleton (lazy-initialized).

    The DB lives inside the workspace's per-UUID directory at
    ``<home_dir>/<workspace_id>/workspace.db`` so it's scoped to the
    workspace, not the install root. Before a workspace exists on
    disk (very first call, ahead of /v1/init or /v1/bootstrap) the
    service runs as a pure proxy without caching; the singleton
    rebinds to the workspace-scoped DB on the next call once a
    workspace has been created.

    Creation is guarded by a lock: this runs as a sync FastAPI dependency
    in the threadpool, so concurrent requests would otherwise race to open
    multiple SQLite connections to the same file.
    """
    global _platform_service, _platform_service_db_path
    from smartloop.config.app_settings import AppSettings
    from smartloop.workspace.workspace import Workspace

    settings = AppSettings()
    ws = Workspace.load()
    db_path = (
        os.path.join(ws.workspace_dir, "workspace.db")
        if ws is not None and ws.workspace_dir
        else None
    )

    # Fast path: already bound to the right DB.
    if _platform_service is not None and _platform_service_db_path == db_path:
        return _platform_service

    with _platform_service_lock:
        # Re-check inside the lock — another thread may have just built it.
        if _platform_service is not None and _platform_service_db_path == db_path:
            return _platform_service

        client = SLPPlatformClient(settings.slp_api_base)
        db = WorkspaceDB(db_path) if db_path else None
        if db is not None and ws is not None:
            db.save_workspace(ws.id)
        _platform_service = PlatformService(client, db)
        _platform_service_db_path = db_path
        return _platform_service
