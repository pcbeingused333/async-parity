from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import scan_paths

DEFAULT_EXCLUDE = ("tests", "test", ".venv", "venv", "node_modules", "build", "dist", ".git")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="async-parity",
        description="Report sync/async method pairs whose bodies diverge beyond the async plumbing.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories to scan")
    parser.add_argument(
        "--all",
        action="store_true",
        help="also report pairs where the sync side has nothing the async side lacks",
    )
    parser.add_argument(
        "--include-tests", action="store_true", help="scan test directories too (skipped by default)"
    )
    parser.add_argument("--max-lines", type=int, default=60, help="diff lines to print per pair (default 60)")
    parser.add_argument("--summary", action="store_true", help="print one line per pair instead of diffs")
    args = parser.parse_args(argv)

    exclude = () if args.include_tests else DEFAULT_EXCLUDE
    found = 0
    for div in scan_paths(args.paths, exclude=exclude):
        # By default only show pairs where the sync body does something the async one doesn't:
        # those are the ones where a fix most likely landed on one side only.
        if not args.all and not div.sync_only:
            continue
        found += 1
        if args.summary:
            print(f"{div.header()}  (+{len(div.async_only)} -{len(div.sync_only)})")
        else:
            print(f"\n### {div.header()}")
            print("\n".join(div.diff[: args.max_lines]))
    print(f"\n{found} divergent pair(s)", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
