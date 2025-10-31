"""Command-line entrypoint to run database migrations for weekly pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from schema_migrations import MigrationManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NFL algorithm database migrations")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(config.database.path),
        help="Path to SQLite database (defaults to config.database.path)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MigrationManager(args.database).run()


if __name__ == "__main__":
    main()
