"""Build per-episode P0 label sidecars.

A label sidecar is a JSONL file where each line corresponds to one episode row
and contains the P0 supervision labels for that row.  Labels are targets only;
they must never be fed into WAM inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from starVLA.dataloader.mowa.full_head_label_builder import _build_episode_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MoWA P0 label sidecars.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Dataset root containing data/chunk-000/episode_*.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for ep_*.jsonl sidecar files.",
    )
    parser.add_argument(
        "--episode-index",
        dest="episode_indices",
        type=int,
        action="append",
        default=None,
        help="Episode index to include. Can be repeated. Default scans all episodes.",
    )
    return parser.parse_args()


def build_label_sidecar_for_episode(
    dataset_path: Path,
    episode_index: int,
) -> tuple[dict[str, Any], ...]:
    """Build one label row per frame for a single episode."""

    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")
    samples = _build_episode_samples(parquet_path, episode_index, preview_rows=1_000_000)
    return tuple(sample.to_dict() for sample in samples)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = dataset_path / "data" / "chunk-000"
    parquet_paths = sorted(data_dir.glob("episode_*.parquet"))

    if args.episode_indices is not None:
        allowed = set(args.episode_indices)
        parquet_paths = [
            path for path in parquet_paths if int(path.stem.split("_")[-1]) in allowed
        ]

    for parquet_path in parquet_paths:
        episode_index = int(parquet_path.stem.split("_")[-1])
        sidecar_path = output_dir / f"ep_{episode_index:06d}.jsonl"
        rows = build_label_sidecar_for_episode(dataset_path, episode_index)
        with sidecar_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows to {sidecar_path}")


if __name__ == "__main__":
    main()
