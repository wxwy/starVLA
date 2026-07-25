#!/usr/bin/env python
"""Visualize MoWA future-label sidecars for one episode.

Examples
--------
    python tools/mowa/visualize_future_labels.py --task CloseBlenderLid --episode 0

    python tools/mowa/visualize_future_labels.py \
        --task OpenDrawer --episode 10 --output /tmp/ovf_check.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MoWA future-label sidecar.")
    parser.add_argument("--task", required=True, help="Atomic task name.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365"),
        help="RoboCasa365 dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: docs_zh/mowa/head_labels_check/<task>_episode_XXXXXX.png",
    )
    return parser.parse_args()


def find_sidecar_path(data_root: Path, task: str, episode_index: int) -> Path | None:
    """Find the sidecar parquet for a given task/episode."""
    atomic_root = data_root / "v1.0" / "target" / "atomic" / task
    if not atomic_root.is_dir():
        return None
    for date_dir in sorted(atomic_root.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.isdigit():
            continue
        candidate = date_dir / "lerobot" / "mowa_future_labels" / task / f"episode_{episode_index:06d}.parquet"
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    sidecar_path = find_sidecar_path(args.data_root, args.task, args.episode)
    if sidecar_path is None:
        raise FileNotFoundError(
            f"No sidecar found for {args.task} episode {args.episode} under {args.data_root}"
        )

    table = pq.read_table(sidecar_path)
    data = table.to_pydict()

    # Build a common time axis.
    frame_index = np.asarray(data.get("frame_index", np.arange(len(data["task_progress"]))), dtype=np.int64)

    # Collect label series and masks.
    label_specs: list[tuple[str, str, str, bool]] = [
        ("task_progress", "task_progress", "", False),
        ("subgoal_feasibility", "subgoal_feasibility", "subgoal_feasibility_mask", True),
        ("manipulation_readiness", "manipulation_readiness", "manipulation_readiness_mask", True),
        ("failure_risk", "failure_risk", "failure_risk_mask", True),
        ("object_visibility_future", "object_visibility_future", "object_visibility_future_mask", True),
        ("next_best_view_score", "next_best_view_score", "next_best_view_score_mask", True),
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(label_specs), 1, figsize=(14, 12), sharex=True)
    if len(label_specs) == 1:
        axes = [axes]

    for ax, (title, value_key, mask_key, has_mask) in zip(axes, label_specs, strict=True):
        values = np.asarray(data[value_key], dtype=np.float64)
        ax.plot(frame_index, values, label=title, linewidth=1.2)

        if has_mask:
            masks = np.asarray(data[mask_key], dtype=bool)
            active = ~masks
            if active.any():
                ax.scatter(
                    frame_index[active],
                    values[active],
                    s=8,
                    c="tab:green",
                    label="active (unmasked)",
                    zorder=3,
                )
            if masks.any():
                ax.axvspan(
                    frame_index[0] if frame_index.size else 0,
                    frame_index[-1] if frame_index.size else 0,
                    alpha=0.1,
                    color="gray",
                    label="masked region",
                )
                masked_indices = frame_index[masks]
                ax.scatter(
                    masked_indices,
                    np.zeros_like(masked_indices),
                    s=4,
                    c="gray",
                    marker="x",
                    label="masked step",
                    zorder=2,
                )

        ax.set_ylabel(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("frame_index")
    fig.suptitle(f"{args.task} episode {args.episode:06d}\n{sidecar_path}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.output is None:
        out_dir = Path("docs_zh/mowa/head_labels_check")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / f"{args.task}_episode_{args.episode:06d}.png"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote visualization to {args.output}")


if __name__ == "__main__":
    main()
