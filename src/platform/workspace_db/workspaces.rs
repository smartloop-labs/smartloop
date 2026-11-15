//! The `workspaces` table: one durable row per workspace, the anchor its
//! projects (and later its members) reference.

use rusqlite::{params, Transaction};

/// Record *workspace_id* if it isn't yet, inside the caller's transaction.
pub(super) fn ensure(tx: &Transaction, workspace_id: &str, created_at: i64) -> anyhow::Result<()> {
    tx.execute(
        "INSERT INTO workspaces (id, created_at) VALUES (?, ?)
           ON CONFLICT(id) DO NOTHING",
        params![workspace_id, created_at],
    )?;
    Ok(())
}
