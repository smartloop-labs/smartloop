//! The `projects` table: one row per project in a workspace.
//!
//! config.json stays the source of the project list; a row here exists so
//! per-project tables have something to reference and cascade from.  Rows
//! are added on create and import, removed on delete, and backfilled from
//! config.json on load, never pruned.

use std::collections::HashSet;

use rusqlite::{params, Transaction};

use super::{workspaces, WorkspaceDB};
use crate::platform::service::now;

/// Queries on the `projects` table ([`WorkspaceDB::projects`]).
pub struct Projects<'a> {
    db: &'a WorkspaceDB,
}

impl WorkspaceDB {
    /// The workspace's projects.
    pub fn projects(&self) -> Projects<'_> {
        Projects { db: self }
    }
}

fn insert(tx: &Transaction, workspace_id: &str, project_id: &str, created_at: i64) -> anyhow::Result<()> {
    tx.execute(
        "INSERT INTO projects (id, workspace_id, created_at) VALUES (?, ?, ?)
           ON CONFLICT(id) DO NOTHING",
        params![project_id, workspace_id, created_at],
    )?;
    Ok(())
}

impl Projects<'_> {
    /// Record *project_id* in *workspace_id*, and the workspace with it.  A
    /// project already recorded keeps its row.
    pub fn add(&self, workspace_id: &str, project_id: &str) -> anyhow::Result<()> {
        let created_at = now() as i64;
        self.db.write(|tx| {
            workspaces::ensure(tx, workspace_id, created_at)?;
            insert(tx, workspace_id, project_id, created_at)
        })
    }

    /// Delete *project_id*'s row, and with it every row that references it.
    /// Returns whether there was one.
    pub fn remove(&self, project_id: &str) -> anyhow::Result<bool> {
        self.db.write(|tx| Ok(tx.execute("DELETE FROM projects WHERE id = ?", params![project_id])? > 0))
    }

    /// Ids recorded for *workspace_id*, in no particular order.
    pub fn ids(&self, workspace_id: &str) -> anyhow::Result<Vec<String>> {
        self.db.read(|conn| {
            let mut stmt = conn.prepare("SELECT id FROM projects WHERE workspace_id = ?")?;
            let ids = stmt
                .query_map(params![workspace_id], |r| r.get(0))?
                .collect::<Result<Vec<String>, _>>()?;
            Ok(ids)
        })
    }

    /// Record any of *project_ids* not recorded yet; rows for projects not
    /// listed are kept.  Writes nothing when every id is already there.
    /// Returns how many were added.
    pub fn backfill<'p>(
        &self,
        workspace_id: &str,
        project_ids: impl IntoIterator<Item = &'p str>,
    ) -> anyhow::Result<usize> {
        let known: HashSet<String> = self.ids(workspace_id)?.into_iter().collect();
        let mut missing: Vec<&str> = project_ids.into_iter().filter(|id| !known.contains(*id)).collect();
        missing.sort_unstable();
        missing.dedup();
        if missing.is_empty() {
            return Ok(0);
        }
        let created_at = now() as i64;
        self.db.write(|tx| {
            workspaces::ensure(tx, workspace_id, created_at)?;
            for project_id in &missing {
                insert(tx, workspace_id, project_id, created_at)?;
            }
            Ok(())
        })?;
        Ok(missing.len())
    }
}
