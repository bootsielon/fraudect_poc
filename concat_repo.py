#!/usr/bin/env python
"""
concat_repo.py  –  dump every *.py file in the repository into one stream
Usage
-----
python concat_repo.py [repo_root]

If *repo_root* is omitted the current working directory is used.

The script writes to stdout, so redirect it to a file if needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".idea",
    ".pytest_cache",
    ".tox",
    ".eggs",
}

DELIM = "=== {rel_path} ==="      # customise if you like


def iter_py_files(root: Path):
    """Yield all *.py files below *root*, skipping EXCLUDE_DIRS."""
    for path in root.rglob("*.py"):
        # skip unwanted directories (fast path: look at any part of the parents)
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def dump_file(path: Path, root: Path):
    """Print delimiter line + file contents to stdout."""
    rel_path = path.relative_to(root).as_posix()
    print(DELIM.format(rel_path=rel_path))
    with path.open("r", encoding="utf-8", errors="replace") as f:
        sys.stdout.write(f.read())
    # always end with exactly one blank line for separation
    sys.stdout.write("\n\n")


def main():
    parser = argparse.ArgumentParser(description="Concatenate all *.py files in a repo")
    parser.add_argument("root", nargs="?", default=".", help="repository root (default: .)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a directory")

    for py_file in sorted(iter_py_files(root)):
        dump_file(py_file, root)


if __name__ == "__main__":
    main()