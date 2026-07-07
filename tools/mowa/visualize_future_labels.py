#!/usr/bin/env python
"""Visualize MoWA future labels overlaid on RGB video frames.

This tool renders a single episode with text overlays showing the per-frame
future labels (progress, subgoal feasibility, manipulation readiness, failure
risk) so that humans can spot-check whether the label construction logic is
reasonable.

Examples
--------
    # Use the default agent-view camera and write to /tmp.
    python tools/mowa/visualize_future_labels.py --task OpenDrawer --episode 0

    # Pick a different camera and output path.
    python tools/mowa/visualize_future_labels.py \\
        --task OpenCabinet \\
        --episode 12 \\
        --camera observation.images.robot0_eye_in_hand \\
        --output /tmp/opencabinet_ep12.mp4

    # Force re-computation even when a sidecar exists.
    python tools/mowa/visualize_future_labels.py --task OpenDrawer --episode 0 --recompute
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from starVLA.dataloader.mowa.atomic_task_label_builder import (
    get_builder_for_task,
    get_task_name_from_dataset_path,
)
from starVLA.dataloader.mowa.label_cache import (
    build_label_cache_for_episode,
    load_label_cache_for_episode,
)


DEFAULT_DATA_ROOT = Path("playground/Datasets/robocasa365")
DEFAULT_CAMERA = "observation.images.robot0_agentview_left"
DEFAULT_OUTPUT_ROOT = Path("/tmp/mowa_future_label_vis")


def find_dataset_path(task_name: str, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    """Locate the LeRobot dataset path for an atomic task."""
    base = data_root / "v1.0" / "target" / "atomic" / task_name
    if not base.exists():
        raise FileNotFoundError(f"Task directory not found: {base}")
    date_dirs = [d for d in sorted(base.iterdir()) if d.is_dir() and re.fullmatch(r"\d{8}", d.name)]
    if not date_dirs:
        raise FileNotFoundError(f"No dated run directory under {base}")
    for date_dir in date_dirs:
        lr = date_dir / "lerobot"
        if lr.exists():
            return lr
    raise FileNotFoundError(f"No lerobot directory found under {base}")


def load_sidecar_or_build(
    builder,
    dataset_path: Path,
    episode_index: int,
    recompute: bool = False,
) -> dict:
    """Load a cached sidecar if available; otherwise build labels on the fly."""
    if not recompute:
        cache = load_label_cache_for_episode(builder, dataset_path, episode_index)
        if cache is not None:
            # Normalize to the same dict layout produced by build_label_cache_for_episode.
            return {
                "frame_index": cache["frame_index"],
                "task_progress": None,  # sidecar may not store progress; we handle below
                "failure_risk": cache["labels"]["failure_risk"],
                "failure_risk_mask": ~cache["masks"]["failure_risk"],
                "subgoal_feasibility": cache["labels"]["subgoal_feasibility"],
                "subgoal_feasibility_mask": ~cache["masks"]["subgoal_feasibility"],
                "manipulation_readiness": cache["labels"]["manipulation_readiness"],
                "manipulation_readiness_mask": ~cache["masks"]["manipulation_readiness"],
                "row_count": len(cache["frame_index"]),
                "source": "sidecar",
            }

    ep_dir = dataset_path / "extras" / f"episode_{episode_index:06d}"
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    cache = build_label_cache_for_episode(
        builder=builder,
        parquet_path=parquet_path,
        states_path=ep_dir / "states.npz",
        model_path=ep_dir / "model.xml.gz",
        ep_meta_path=ep_dir / "ep_meta.json",
        enable_kinematics=True,
    )
    cache["source"] = "built"
    return cache


def _progress_bar(frame: np.ndarray, progress: float, y: int, width: int = 200, height: int = 16) -> None:
    """Draw a horizontal progress bar on the frame."""
    x = 10
    filled = int(np.clip(progress, 0.0, 1.0) * width)
    # background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    # fill
    cv2.rectangle(frame, (x, y), (x + filled, y + height), (0, 200, 0), -1)
    # border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    # text
    cv2.putText(frame, f"{progress:.2f}", (x + width + 8, y + height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _state_badge(frame: np.ndarray, label: str, value: float, masked: bool,
                 y: int, active_color: tuple[int, int, int]) -> None:
    """Draw a colored badge for a binary label."""
    x = 10
    text = f"{label}: {'ON' if value > 0.5 else 'off'}"
    if masked:
        text += " (masked)"
        color = (80, 80, 80)
    else:
        color = active_color if value > 0.5 else (40, 40, 40)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y), (x + tw + 10, y + th + 8), color, -1)
    cv2.putText(frame, text, (x + 5, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_overlay(frame: np.ndarray, cache: dict, row: int, task_name: str,
                  camera_name: str, fps: float) -> np.ndarray:
    """Draw label overlay on a single frame."""
    h, w = frame.shape[:2]
    # Dark header background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    progress = float(cache.get("task_progress", [0.0] * (row + 1))[row]) if cache.get("task_progress") is not None else 0.0
    sf = float(cache["subgoal_feasibility"][row])
    sf_mask = bool(cache["subgoal_feasibility_mask"][row])
    mr = float(cache["manipulation_readiness"][row])
    mr_mask = bool(cache["manipulation_readiness_mask"][row])
    fr = float(cache["failure_risk"][row])
    fr_mask = bool(cache["failure_risk_mask"][row])

    header = f"{task_name} | ep {cache['frame_index'][row]:06d} | {camera_name} | fps {fps:.1f} | src {cache.get('source', 'unknown')}"
    cv2.putText(frame, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    _progress_bar(frame, progress, y=32)
    _state_badge(frame, "subgoal", sf, sf_mask, y=56, active_color=(0, 165, 255))
    _state_badge(frame, "ready", mr, mr_mask, y=78, active_color=(0, 255, 0))
    _state_badge(frame, "risk", fr, fr_mask, y=100, active_color=(0, 0, 255))

    return frame


def visualize(
    task_name: str,
    episode_index: int,
    camera_name: str = DEFAULT_CAMERA,
    output_path: Path | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    recompute: bool = False,
) -> Path:
    """Render one episode with label overlay and return the output path."""
    dataset_path = find_dataset_path(task_name, data_root)
    builder = get_builder_for_task(task_name)
    cache = load_sidecar_or_build(builder, dataset_path, episode_index, recompute=recompute)

    video_dir = dataset_path / "videos" / "chunk-000" / camera_name
    video_path = video_dir / f"episode_{episode_index:06d}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path is None:
        output_root = DEFAULT_OUTPUT_ROOT
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{task_name}_ep{episode_index:06d}_{camera_name.replace('.', '_')}.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    row = 0
    total_rows = int(cache["row_count"])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if row >= total_rows:
            # More video frames than labels; stop.
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _draw_overlay(frame, cache, row, task_name, camera_name, fps)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
        row += 1

    cap.release()
    writer.release()
    print(f"Wrote {row} frames to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MoWA future labels on RGB video.")
    parser.add_argument("--task", required=True, help="Atomic task name, e.g. OpenDrawer.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index.")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="Camera/video stream name.")
    parser.add_argument("--output", default=None, help="Output MP4 path.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to robocasa365 dataset root.")
    parser.add_argument("--recompute", action="store_true", help="Recompute labels instead of reading sidecar.")
    args = parser.parse_args()

    output_path = visualize(
        task_name=args.task,
        episode_index=args.episode,
        camera_name=args.camera,
        output_path=Path(args.output) if args.output else None,
        data_root=args.data_root,
        recompute=args.recompute,
    )
    print(output_path)


if __name__ == "__main__":
    main()
