"""Asset Pipeline CLI.

Usage (run from the asset_pipeline/ directory, or `python -m asset_pipeline.cli`
from the repo root once __main__ wiring is added in a later stage):
    python cli.py catalog list
    python cli.py catalog seed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from pipeline.catalog_writer import load_catalog
from pipeline.seed_catalog import main as seed_catalog_main


def cmd_catalog_list(args: argparse.Namespace) -> int:
    entries = load_catalog()
    if not entries:
        print(f"Catalog is empty ({config.CATALOG_PATH}). Run `catalog seed` first.")
        return 0
    print(f"{'asset_id':<20} {'class':<11} {'review':<7} {'bounds_m':<20} tags")
    print("-" * 90)
    for e in entries:
        bounds = "x".join(f"{v:g}" for v in e.canonical_bounds_m)
        tags = ", ".join(e.tags[:5])
        print(f"{e.asset_id:<20} {e.asset_class:<11} {e.review_level:<7} {bounds:<20} {tags}")
    print(f"\n{len(entries)} asset(s) in {config.CATALOG_PATH}")
    return 0


def cmd_catalog_seed(args: argparse.Namespace) -> int:
    seed_catalog_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asset_pipeline",
        description="Offline AR asset acquisition pipeline CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="Inspect or seed the asset catalog")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_sub.add_parser("list", help="List all catalog entries").set_defaults(func=cmd_catalog_list)
    catalog_sub.add_parser(
        "seed", help="(Re)write the seed catalog of core primitives"
    ).set_defaults(func=cmd_catalog_seed)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
