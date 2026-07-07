#!/usr/bin/env python
"""Visualize MoWA future labels overlaid on RGB video frames.

This tool renders a single episode with text overlays showing the per-frame
future labels (progress, subgoal feasibility, manipulation readiness, failure
risk) so that humans can spot-check whether the label construction logic is
reasonable.

By default it concatenates the agent-view camera with the eye-in-hand camera
horizontally and writes the result to
``playground/head_labels_check/<task>_ep<idx>.mp4``.

Examples
--------
    # Use the default agent-view + wrist cameras.
    python tools/mowa/visualize_future_labels.py --task OpenDrawer --episode 0

    # Single camera only.
    python tools/mowa/visualize_future_labels.py --task OpenDrawer --episode 0 --no-wrist

    # Force re-computation even when a sidecar exists.
    python tools/mowa/visualize_future_labels.py --task OpenDrawer --episode 0 --recompute
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from starVLA.dataloader.mowa.atomic_task_label_builder import (
    get_builder_for_task,
)
from starVLA.dataloader.mowa.label_cache import (
    build_label_cache_for_episode,
    load_label_cache_for_episode,
)


DEFAULT_DATA_ROOT = Path("playground/Datasets/robocasa365")
DEFAULT_AGENT_CAMERA = "observation.images.robot0_agentview_left"
DEFAULT_WRIST_CAMERA = "observation.images.robot0_eye_in_hand"
DEFAULT_OUTPUT_ROOT = Path("playground/head_labels_check")

# UI scale: smaller than before so the overlay takes ~50% less screen real-estate.
FONT_SCALE = 0.25
TEXT_THICKNESS = 1
LINE_HEIGHT = 12
HEADER_HEIGHT = 54
BAR_HEIGHT = 6
BADGE_HEIGHT = 10


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
            return {
                "frame_index": cache["frame_index"],
                "task_progress": None,
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


def _progress_bar(frame: np.ndarray, progress: float, y: int, x: int = 6,
                  width: int = 80, height: int = BAR_HEIGHT) -> None:
    """Draw a horizontal progress bar on the frame."""
    filled = int(np.clip(progress, 0.0, 1.0) * width)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + height), (0, 200, 0), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    cv2.putText(frame, f"{progress:.2f}", (x + width + 4, y + height),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)


def _state_badge(frame: np.ndarray, label: str, value: float, masked: bool,
                 y: int, active_color: tuple[int, int, int], x: int = 6) -> None:
    """Draw a colored badge for a binary label."""
    text = f"{label}: {'ON' if value > 0.5 else 'off'}"
    if masked:
        text += " (masked)"
        color = (80, 80, 80)
    else:
        color = active_color if value > 0.5 else (40, 40, 40)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)
    cv2.rectangle(frame, (x, y), (x + tw + 4, y + th + 3), color, -1)
    cv2.putText(frame, text, (x + 2, y + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)


def _draw_overlay(frame: np.ndarray, cache: dict, row: int, task_name: str,
                  camera_text: str, fps: float) -> np.ndarray:
    """Draw label overlay on a single frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, HEADER_HEIGHT), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    progress = float(cache.get("task_progress", [0.0] * (row + 1))[row]) if cache.get("task_progress") is not None else 0.0
    sf = float(cache["subgoal_feasibility"][row])
    sf_mask = bool(cache["subgoal_feasibility_mask"][row])
    mr = float(cache["manipulation_readiness"][row])
    mr_mask = bool(cache["manipulation_readiness_mask"][row])
    fr = float(cache["failure_risk"][row])
    fr_mask = bool(cache["failure_risk_mask"][row])

    header = f"{task_name} | ep {cache['frame_index'][row]:06d} | {camera_text} | {fps:.1f}fps"
    cv2.putText(frame, header, (6, 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA)

    _progress_bar(frame, progress, y=16)
    _state_badge(frame, "subgoal", sf, sf_mask, y=26, active_color=(0, 165, 255))
    _state_badge(frame, "ready", mr, mr_mask, y=38, active_color=(0, 255, 0))
    _state_badge(frame, "risk", fr, fr_mask, y=50, active_color=(0, 0, 255))

    return frame


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def visualize(
    task_name: str,
    episode_index: int,
    agent_camera: str = DEFAULT_AGENT_CAMERA,
    wrist_camera: str | None = DEFAULT_WRIST_CAMERA,
    output_path: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    recompute: bool = False,
) -> Path:
    """Render one episode with label overlay and return the output path."""
    dataset_path = find_dataset_path(task_name, data_root)
    builder = get_builder_for_task(task_name)
    cache = load_sidecar_or_build(builder, dataset_path, episode_index, recompute=recompute)

    agent_video_path = dataset_path / "videos" / "chunk-000" / agent_camera / f"episode_{episode_index:06d}.mp4"
    if not agent_video_path.is_file():
        raise FileNotFoundError(f"Agent video not found: {agent_video_path}")
    agent_cap = _open_video(agent_video_path)

    wrist_cap: cv2.VideoCapture | None = None
    wrist_video_path: Path | None = None
    if wrist_camera is not None:
        wrist_video_path = dataset_path / "videos" / "chunk-000" / wrist_camera / f"episode_{episode_index:06d}.mp4"
        if wrist_video_path.is_file():
            wrist_cap = _open_video(wrist_video_path)
        else:
            wrist_video_path = None
            wrist_camera = None

    fps = agent_cap.get(cv2.CAP_PROP_FPS) or 10.0
    agent_width = int(agent_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    agent_height = int(agent_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if wrist_cap is not None:
        wrist_width = int(wrist_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        wrist_height = int(wrist_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (wrist_width, wrist_height) != (agent_width, agent_height):
            raise ValueError(
                f"Camera size mismatch: agent={agent_width}x{agent_height}, "
                f"wrist={wrist_width}x{wrist_height}"
            )
        out_width = agent_width + wrist_width
        out_height = agent_height
        camera_text = "agent+wrist"
    else:
        out_width = agent_width
        out_height = agent_height
        camera_text = agent_camera.split(".")[-1]

    if output_path is None:
        output_root.mkdir(parents=True, exist_ok=True)
        suffix = "_agent_wrist" if wrist_cap is not None else "_agent"
        output_path = output_root / f"{task_name}_ep{episode_index:06d}{suffix}.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    row = 0
    total_rows = int(cache["row_count"])
    while True:
        ret_agent, agent_frame = agent_cap.read()
        if not ret_agent:
            break

        if wrist_cap is not None:
            ret_wrist, wrist_frame = wrist_cap.read()
            if not ret_wrist:
                break
        else:
            wrist_frame = None

        if row >= total_rows:
            break

        agent_frame = cv2.cvtColor(agent_frame, cv2.COLOR_BGR2RGB)
        agent_frame = _draw_overlay(agent_frame, cache, row, task_name, camera_text, fps)
        agent_frame = cv2.cvtColor(agent_frame, cv2.COLOR_RGB2BGR)

        if wrist_frame is not None:
            wrist_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
            wrist_frame = _draw_overlay(wrist_frame, cache, row, task_name, "wrist", fps)
            wrist_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
            combined = np.concatenate([agent_frame, wrist_frame], axis=1)
        else:
            combined = agent_frame

        writer.write(combined)
        row += 1

    agent_cap.release()
    if wrist_cap is not None:
        wrist_cap.release()
    writer.release()
    print(f"Wrote {row} frames to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MoWA future labels on RGB video.")
    parser.add_argument("--task", required=True, help="Atomic task name, e.g. OpenDrawer.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index.")
    parser.add_argument("--agent-camera", default=DEFAULT_AGENT_CAMERA, help="Agent/external camera stream name.")
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA, help="Wrist/eye-in-hand camera stream name.")
    parser.add_argument("--no-wrist", action="store_true", help="Disable wrist camera concatenation.")
    parser.add_argument("--output", default=None, help="Output MP4 path.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Default output directory.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to robocasa365 dataset root.")
    parser.add_argument("--recompute", action="store_true", help="Recompute labels instead of reading sidecar.")
    args = parser.parse_args()

    visualize(
        task_name=args.task,
        episode_index=args.episode,
        agent_camera=args.agent_camera,
        wrist_camera=None if args.no_wrist else args.wrist_camera,
        output_path=Path(args.output) if args.output else None,
        output_root=args.output_root,
        data_root=args.data_root,
        recompute=args.recompute,
    )


if __name__ == "__main__":
    main()
