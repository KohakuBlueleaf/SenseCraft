"""Release notes for a version: its Change.md section, or the commit list.

Change.md is the curated source. When it has no section for the version being released (a patch
cut by automation, say), the notes fall back to the commits since the previous tag, so a release
is never published with empty notes.

Usage:
    python scripts/ci/release_notes.py --version 0.4.0
"""

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = ROOT / "Change.md"


def changelog_section(version: str) -> str | None:
    """The body under `## ... update to <version>`, up to the next `## ` heading."""
    if not CHANGELOG.exists():
        return None
    text = CHANGELOG.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(?ms)^##\s+[^\n]*update to {re.escape(version)}\s*$(.*?)(?=^##\s|\Z)"
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def commits_since_last_tag() -> str:
    """One line per commit since the previous tag."""
    tags = subprocess.run(
        ["git", "tag", "--sort=-creatordate"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    ).stdout.split()
    span = f"{tags[1]}..HEAD" if len(tags) > 1 else "HEAD"
    log = subprocess.run(
        ["git", "log", "--no-merges", "--pretty=* %s (%h)", span],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    ).stdout.strip()
    return log or "* No changes recorded."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    body = changelog_section(args.version)
    if body is None:
        body = f"### Changes\n\n{commits_since_last_tag()}"
    print(f"## SenseCraft {args.version}\n\n{body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
