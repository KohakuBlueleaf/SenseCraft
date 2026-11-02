"""Read, stamp and bump the version: `[project] version` in pyproject.toml is the source of truth.

Usage:
    python scripts/ci/version.py read
    python scripts/ci/version.py nightly            # X.Y.(Z+1).devYYYYmmddHHMMSS
    python scripts/ci/version.py bump patch|minor|major
    python scripts/ci/version.py write <version>    # stamp pyproject.toml
"""

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
# The first `version = "..."` line: [project] comes before any other table that has one
PATTERN = re.compile(r'(?m)^(version\s*=\s*")([^"]+)(")')


def read() -> str:
    """The version in pyproject.toml."""
    match = PATTERN.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("no version found in pyproject.toml")
    return match.group(2)


def write(version: str) -> None:
    """Stamp `version` into pyproject.toml."""
    text = PYPROJECT.read_text(encoding="utf-8")
    new, count = PATTERN.subn(rf"\g<1>{version}\g<3>", text, count=1)
    if count != 1:
        raise SystemExit("no version found in pyproject.toml")
    PYPROJECT.write_text(new, encoding="utf-8")


def base(version: str) -> tuple[int, int, int]:
    """Major, minor and patch of a PEP 440 version."""
    core = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not core:
        raise SystemExit(f"cannot parse version {version!r}")
    major, minor, patch = (int(group) for group in core.groups())
    return major, minor, patch


def bump(version: str, part: str) -> str:
    """The next `part` release after `version`."""
    major, minor, patch = base(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def nightly(version: str) -> str:
    """PEP 440 dev release of the next patch, ordered by its UTC timestamp."""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    major, minor, patch = base(version)
    return f"{major}.{minor}.{patch + 1}.dev{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("read", "nightly", "bump", "write"))
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    current = read()

    if args.action == "read":
        print(current)
    elif args.action == "nightly":
        print(nightly(current))
    elif args.action == "bump":
        print(bump(current, args.value or "patch"))
    else:
        if not args.value:
            raise SystemExit("write needs a version")
        write(args.value)
        print(args.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
