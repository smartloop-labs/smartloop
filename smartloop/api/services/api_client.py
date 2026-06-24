"""Smartloop API client for token usage telemetry."""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from smartloop.platform import SLPPlatformClient
from smartloop.platform import auth as platform_auth
from smartloop.config.app_settings import AppSettings


settings = AppSettings()

logger = logging.getLogger(__name__)

_SLP_DIR = Path(settings.home_dir)
_TOKEN_USAGE_FILE = os.path.join(_SLP_DIR, "token_usage.jsonl")


class SLPAPIClient:
    """HTTP client for pushing telemetry to the Smartloop platform API.

    Uses the developer token set during ``slp init`` for authentication.
    Self-disables gracefully when no token is configured so the rest of
    the application continues to work without an API connection.

    Every public method catches all exceptions — the client never crashes
    the host process.
    """

    def __init__(self, settings: Optional[AppSettings] = None):
        self._settings = settings or AppSettings()
        self._platform_client = SLPPlatformClient(self._settings.slp_api_base)

    @property
    def enabled(self) -> bool:
        return platform_auth.has_credentials()

    # -------------------------------------------------------------- Telemetry

    def push_token_usage(
        self,
        *,
        project_id: Optional[str],
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record token usage locally and push to the Smartloop API.

        Runs the HTTP POST on a daemon thread so it never blocks the
        caller.  Local append to ``~/.slp/token_usage.jsonl`` always
        happens regardless of API availability.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_id": project_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        # Always persist locally
        try:
            _SLP_DIR.mkdir(parents=True, exist_ok=True)
            with open(_TOKEN_USAGE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write local token usage: {e}")

        if not self.enabled:
            return

        def _post():
            try:
                self._platform_client.push_token_usage(record)
                logger.info("Token usage pushed (%d tokens, project=%s)", record["total_tokens"], record.get("project_id"))
            except Exception as e:
                logger.debug(f"Token usage push failed (non-fatal): {e}")

        thread = threading.Thread(target=_post, daemon=True)
        thread.start()
