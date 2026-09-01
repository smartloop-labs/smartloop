"""The picker reads and writes the thinking setting through /v1/models.

The toggle has to be visible per model (so the UI knows which models can
reason) and settable (so switching it on sticks) — and switching it on has to
reach a model that is already loaded, since the always-resident orchestrator
would otherwise never pick the change up.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.api.routers.models import _UpdateWorkspaceModelRequest, update_workspace_model
from smartloop.workspace.workspace import Project, Workspace, WorkspaceModel
from fastapi import HTTPException


def _workspace() -> Workspace:
    return Workspace(
        id="ws1",
        models=[
            WorkspaceModel(model_name="qwen3.5-2b", model_id="Qwen/Qwen3.5-2B",
                           downloaded=True, enabled=True),
            WorkspaceModel(model_name="gemma3-4b", model_id="google/gemma-3-4b-it",
                           downloaded=True, enabled=True),
        ],
        projects=[Project(id="p1", project_dir="/tmp/p1", current=True, name="p1")],
    )


def _service(workspace):
    svc = MagicMock()
    svc._ensure_workspace.return_value = workspace
    return svc


@pytest.fixture
def saves(monkeypatch):
    """Record ``Workspace.save`` calls (it is a pydantic model, not mockable)."""
    calls = []
    monkeypatch.setattr(Workspace, "save", lambda self: calls.append(self))
    return calls


@pytest.mark.asyncio
async def test_a_model_that_cannot_reason_is_rejected(saves):
    # Storing the preference would leave the picker showing a toggle that
    # silently does nothing.
    ws = _workspace()
    svc = _service(ws)

    with pytest.raises(HTTPException) as exc:
        await update_workspace_model(
            "gemma3-4b", _UpdateWorkspaceModelRequest(thinking=True), svc,
        )

    assert exc.value.status_code == 400
    assert ws.get_model("gemma3-4b").thinking is None

