#!/usr/bin/env python3
"""
Update CITATION.cff with current package version and optional date.
Run from repo root. Used in CI/release pipeline so built artifacts have correct metadata.

  python scripts/update_citation_cff.py
  python scripts/update_citation_cff.py --version 0.3.0 --date 2026-02-27
"""
from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path


def get_version_from_scm(root: Path) -> str:
    try:
        from setuptools_scm import get_version
        return get_version(root=root)
    except Exception:
        return "0.0.0+unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update CITATION.cff version and date-released")
    parser.add_argument("--version", default=None, help="Set version (default: from setuptools_scm)")
    parser.add_argument("--date", default=None, help="Set date-released YYYY-MM-DD (default: today)")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repo root (default: parent of scripts/)")
    args = parser.parse_args()

    root = args.repo_root or Path(__file__).resolve().parent.parent
    cff_path = root / "CITATION.cff"
    if not cff_path.exists():
        raise SystemExit(f"CITATION.cff not found at {cff_path}")

    version = args.version or get_version_from_scm(root)
    # Strip local suffix when auto-detecting so CITATION.cff has a clean version (e.g. 0.3.0+g123 -> 0.3.0)
    if not args.version and "+" in version:
        version = version.split("+")[0]
    date_str = args.date or date.today().isoformat()

    text = cff_path.read_text()
    text = re.sub(r"^version:.*$", f'version: "{version}"', text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^date-released:.*$", f"date-released: {date_str}", text, count=1, flags=re.MULTILINE)
    cff_path.write_text(text)
    print(f"Updated {cff_path}: version={version}, date-released={date_str}")


if __name__ == "__main__":
    main()
