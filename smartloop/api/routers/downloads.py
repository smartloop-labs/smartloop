"""Model downloads as something a client can address while they run.

Only the cancel lives here today — a download is still *started* by
``POST /v1/init``, which does nine other things besides.  The route is shaped
and named for where it is going rather than for where it came from, so the
desktop app's stop button is written once: when downloads become their own
resource, ``POST``/``GET`` join this router and this handler does not move.

Workspace-level on purpose.  ``supervisor.routing.project_to_proxy`` forwards a
request that names a project in its path *or its query string*, and the cancel
registry is process-local — a proxied cancel would signal a child's empty
registry while the parent held the running transfer.  So this path names no
project, and must not grow a ``project_id`` parameter.
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from smartloop.api.services import ModelService, get_model_service
from smartloop.config import SUPPORTED_MODELS
from smartloop.constants import canonical_model_name
from smartloop.model.downloader import cancellation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["downloads"])

#: Upper bound on the optional synchronous wait.  Long enough to cover the
#: usual case — threads stop at the next 1-8MB chunk boundary and the tree is
#: removed — without letting a caller hold a worker for the 60s a stalled
#: socket can take (``_STREAM_TIMEOUT.read``).
_MAX_WAIT_MS = 5000


def _resolve_model_id(model_name: str, model_service: ModelService) -> str:
    """The ``model_id`` naming *model_name*'s directory, or raise ``ValueError``.

    The workspace record first, because that is what the download is actually
    writing under; ``SUPPORTED_MODELS`` covers a cancel that races the
    registration in ``/v1/init``.
    """
    workspace = model_service._ensure_workspace()
    entry = workspace.get_model(model_name)
    if entry is not None and entry.model_id:
        return entry.model_id

    model_cls = SUPPORTED_MODELS.get(model_name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {model_name}")
    return model_cls.model_id()


@router.delete("/downloads/{model_name}")
async def cancel_download(
    model_name: str,
    wait_ms: int = Query(default=0, ge=0, description="Block up to this long for the bytes to be gone"),
    model_service: ModelService = Depends(get_model_service),
):
    """Stop the download of *model_name* and discard what it wrote.

    Signals and returns.  It does not delete anything itself and it does not
    wait by default: cleanup belongs to the download thread, which is the only
    one that knows every worker has stopped, and a route that unlinked while
    ``_download_file`` was mid-``iter_bytes`` would race an open write.  A
    blocking cancel is also what burned the desktop side the first time.

    A download that is already finished, or was never running, is a 200 with
    ``cancelled: false`` rather than a 404.  The client races by construction —
    a progress frame arrives, the user clicks, the transfer completes before the
    request lands — and answering 404 would push the picker into an error path
    for an ordinary outcome.  The flag tells it to re-sync from
    ``GET /v1/models`` instead.

    ``wait_ms`` trades that promptness for certainty: it waits (bounded) for the
    transfer's threads to stop *and* its bytes to be discarded, and reports
    ``stopped`` when both have happened.  For the CLI, and for tests that would
    otherwise sleep on a guess.
    """
    try:
        # An older client may still address the model by its previous name.
        name = canonical_model_name(model_name)
        model_id = _resolve_model_id(name, model_service)

        token = cancellation.request_cancel(model_id)
        if token is None:
            return {"cancelled": False, "status": "not_found",
                    "model_name": name, "model_id": model_id}

        status = "cancelling"
        if wait_ms:
            timeout = min(wait_ms, _MAX_WAIT_MS) / 1000.0
            loop = asyncio.get_event_loop()
            if await loop.run_in_executor(None, token.done.wait, timeout):
                status = "stopped"

        logger.info("[downloads] Cancel requested for %s (%s)", name, status)
        return {"cancelled": True, "status": status,
                "model_name": name, "model_id": model_id}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error cancelling download for %s", model_name)
        raise HTTPException(status_code=500, detail=str(e))
