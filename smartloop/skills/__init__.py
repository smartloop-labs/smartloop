"""Built-in system skill templates.

When a new workspace is created the *.md files in this package directory
are copied into ``<workspace_dir>/skills/`` so they are auto-loaded as
system skills on every chat request.
"""

import os
import shutil
from pathlib import Path

SKILLS_TEMPLATE_DIR = Path(__file__).parent


def install_default_skills(dest_dir: str) -> None:
    """Copy bundled *.md skill files into *dest_dir* (skip existing)."""
    os.makedirs(dest_dir, exist_ok=True)
    for src in SKILLS_TEMPLATE_DIR.glob("*.md"):
        dst = os.path.join(dest_dir, src.name)
        if not os.path.exists(dst):
            shutil.copy2(str(src), dst)
