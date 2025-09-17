#!/usr/bin/env python3
# numpydoc ignore=GL08
"""
Script to clean up generated markdown files that correspond to existing qmd files.

This script recursively searches for .qmd files in a directory and deletes any
corresponding .md files with the same basename. This is useful for cleaning up
markdown files that may have been generated from Quarto documents before
re-rendering.
"""

import argparse
import sys
from pathlib import Path


def cleanup_generated_md(directory: Path, dry_run: bool = False) -> None:
    """
    Clean up generated markdown files corresponding to qmd files.

    Parameters
    ----------
    directory
        The directory to search recursively for qmd files
    dry_run
        If True, only print what would be deleted without actually deleting
    """
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        return

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return

    deleted_count = 0

    # Find all .qmd files recursively
    for qmd_file in directory.rglob("*.qmd"):
        # Construct corresponding .md file path
        md_file = qmd_file.with_suffix(".md")

        if md_file.exists():
            if dry_run:
                print(f"Would delete: {md_file}")
            else:
                try:
                    md_file.unlink()
                    print(f"Deleted: {md_file}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {md_file}: {e}", file=sys.stderr)

    if not dry_run:
        print(f"Cleanup complete. Deleted {deleted_count} markdown files.")
    else:
        print(f"Dry run complete. Would delete {deleted_count} markdown files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up generated markdown files corresponding to qmd files"
    )
    parser.add_argument(
        "directory", type=Path, help="Directory to search recursively for qmd files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    cleanup_generated_md(args.directory, args.dry_run)
