//! Built-in system skill templates.
//!
//! Port of `smartloop/skills/__init__.py`.  When a new workspace is created
//! the bundled `*.md` files are copied into `<workspace_dir>/skills/` so they
//! are auto-loaded as system skills on every chat request.  The templates are
//! embedded in the binary.

use std::path::Path;

/// The bundled skill templates as `(file name, contents)`.
pub const DEFAULT_SKILLS: &[(&str, &str)] = &[(
    "__system_mcp__.md",
    include_str!("__system_mcp__.md"),
)];

/// Copy bundled `*.md` skill files into `dest_dir` (skip existing).
pub fn install_default_skills(dest_dir: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dest_dir)?;
    for (name, contents) in DEFAULT_SKILLS {
        let dst = Path::new(dest_dir).join(name);
        if !dst.exists() {
            std::fs::write(&dst, contents)?;
        }
    }
    Ok(())
}
