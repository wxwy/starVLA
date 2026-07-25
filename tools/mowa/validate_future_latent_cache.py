"""Validate a MoWA future latent cache manifest and artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa.latent_cache_builder import validate_mowa_latent_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MoWA future latent cache.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Cache root containing manifest.json and *.pt artifacts.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest path override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. When omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_mowa_latent_cache(
        args.cache_root,
        manifest_path=args.manifest_path,
    ).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

