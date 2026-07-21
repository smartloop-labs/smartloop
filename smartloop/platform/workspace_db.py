"""Workspace-level SQLite persistence for cached cloud data.

Stores user profile and model catalog locally so the API can serve
responses when the cloud platform is unreachable.  Lives at
``~/.smartloop/workspace.db``.
"""

import json
import os
import sqlite3
import threading
import time
from typing import Optional


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS workspace (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id                  TEXT PRIMARY KEY DEFAULT 'me',
    email               TEXT NOT NULL,
    name                TEXT,
    is_premium          INTEGER NOT NULL DEFAULT 0,
    is_paid             INTEGER NOT NULL DEFAULT 0,
    tokens_processed    INTEGER NOT NULL DEFAULT 0,
    default_temperature REAL NOT NULL DEFAULT 0.3,
    default_memory      INTEGER NOT NULL DEFAULT 1,
    fetched_at          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_catalog (
    model_id      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    size          TEXT,
    description   TEXT,
    access_level  TEXT,
    is_accessible INTEGER NOT NULL DEFAULT 0,
    sha256        TEXT,
    fetched_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS release_cache (
    channel     TEXT PRIMARY KEY,
    release     TEXT,            -- JSON of the latest published ReleaseRead, or NULL
    fetched_at  REAL NOT NULL
);
"""


class WorkspaceDB:
    """Simple SQLite store for workspace-level cached data."""

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def get_user(self) -> Optional[dict]:
        """Return the cached user profile, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE id = 'me'"
            ).fetchone()
        if row is None:
            return None
        return {
            "email": row["email"],
            "name": row["name"],
            "is_premium": bool(row["is_premium"]),
            "is_paid": bool(row["is_paid"]),
            "number_of_tokens_processed": row["tokens_processed"],
            "default_temperature": row["default_temperature"],
            "default_memory": bool(row["default_memory"]),
        }

    def save_user(self, data: dict) -> None:
        """Upsert the user profile from a cloud API response."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO users
                       (id, email, name, is_premium, is_paid, tokens_processed,
                        default_temperature, default_memory, fetched_at)
                   VALUES ('me', ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       email = excluded.email,
                       name = excluded.name,
                       is_premium = excluded.is_premium,
                       is_paid = excluded.is_paid,
                       tokens_processed = excluded.tokens_processed,
                       default_temperature = excluded.default_temperature,
                       default_memory = excluded.default_memory,
                       fetched_at = excluded.fetched_at
                """,
                (
                    data["email"],
                    data.get("name"),
                    int(data.get("is_premium", False)),
                    int(data.get("is_paid", False)),
                    data.get("number_of_tokens_processed", 0),
                    data.get("default_temperature", 0.3),
                    int(data.get("default_memory", True)),
                    time.time(),
                ),
            )
            self._conn.commit()

    def user_fetched_at(self) -> Optional[float]:
        """Return the timestamp of the last successful cloud fetch, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fetched_at FROM users WHERE id = 'me'"
            ).fetchone()
        return row["fetched_at"] if row else None

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return the cached model catalog."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM model_catalog").fetchall()
        return [
            {
                "model_id": r["model_id"],
                "name": r["name"],
                "size": r["size"],
                "description": r["description"],
                "access_level": r["access_level"],
                "is_accessible": bool(r["is_accessible"]),
                "sha256": r["sha256"],
            }
            for r in rows
        ]

    def save_models(self, models: list[dict]) -> None:
        """Replace the entire model catalog from a cloud API response."""
        now = time.time()
        with self._lock:
            self._conn.execute("DELETE FROM model_catalog")
            self._conn.executemany(
                """INSERT INTO model_catalog
                       (model_id, name, size, description, access_level, is_accessible, sha256, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        m["model_id"],
                        m["name"],
                        m.get("size"),
                        m.get("description"),
                        m.get("access_level"),
                        int(m.get("is_accessible", False)),
                        m.get("sha256"),
                        now,
                    )
                    for m in models
                ],
            )
            self._conn.commit()

    def models_fetched_at(self) -> Optional[float]:
        """Return the timestamp of the last model catalog fetch, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(fetched_at) AS ts FROM model_catalog"
            ).fetchone()
        return row["ts"] if row and row["ts"] is not None else None

    # ------------------------------------------------------------------
    # Releases
    # ------------------------------------------------------------------

    def get_release(self, channel: str = "stable") -> Optional[dict]:
        """Return the cached latest release for *channel*, or ``None``.

        ``None`` is returned both when nothing has been cached and when the
        cached value records that the channel has no published release. To tell
        those two states apart, use :meth:`release_fetched_at` — it returns a
        timestamp once the channel has been polled, even if the release is
        ``None``.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT release FROM release_cache WHERE channel = ?", (channel,)
            ).fetchone()
        if row is None or row["release"] is None:
            return None
        return json.loads(row["release"])

    def save_release(self, channel: str, release: Optional[dict]) -> None:
        """Upsert the latest release for *channel* from a cloud API response.

        *release* may be ``None`` to record that the channel currently has no
        published release (still bumps ``fetched_at``).
        """
        with self._lock:
            self._conn.execute(
                """INSERT INTO release_cache (channel, release, fetched_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(channel) DO UPDATE SET
                       release = excluded.release,
                       fetched_at = excluded.fetched_at
                """,
                (channel, json.dumps(release) if release is not None else None, time.time()),
            )
            self._conn.commit()

    def release_fetched_at(self, channel: str = "stable") -> Optional[float]:
        """Return the timestamp of the last release fetch for *channel*, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fetched_at FROM release_cache WHERE channel = ?", (channel,)
            ).fetchone()
        return row["fetched_at"] if row else None

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def save_workspace(self, workspace_id: str) -> None:
        """Record this workspace's id. Inserts on first call; bumps ``updated_at`` on subsequent calls."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO workspace (id, created_at, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       updated_at = excluded.updated_at
                """,
                (workspace_id, now, now),
            )
            self._conn.commit()

    def get_workspace(self) -> Optional[dict]:
        """Return the workspace row, or ``None`` if no workspace has been recorded."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, created_at, updated_at FROM workspace LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
