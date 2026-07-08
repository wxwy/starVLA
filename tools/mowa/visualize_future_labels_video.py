#!/usr/bin/env python
"""Visualize MoWA future-label sidecars as a two-camera video with indicator lights.

For each frame of an episode, the script composites:
- Top: a 2x3 grid of bright-dot indicators, one for each of the six future heads.
- Bottom: side-by-side ``robot0_agentview_left`` and ``robot0_eye_in_hand`` videos.

It also writes a high-resolution static curve plot with the same base filename as
the video (``<task>_episode_XXXXXX_labels.png``).

Examples
--------
    python tools/mowa/visualize_future_labels_video.py --task OpenDrawer --episode 0

    python tools/mowa/visualize_future_labels_video.py \
        --task OpenDrawer --episode 10 \
        --output /tmp/ovf_check.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

LEFT_CAMERA = "robot0_agentview_left"
WRIST_CAMERA = "robot0_eye_in_hand"
LABEL_SPECS: list[tuple[str, str, str | None]] = [
    ("task_progress", "task_progress", None),
    ("subgoal_feasibility", "subgoal_feasibility", "subgoal_feasibility_mask"),
    ("manipulation_readiness", "manipulation_readiness", "manipulation_readiness_mask"),
    ("failure_risk", "failure_risk", "failure_risk_mask"),
    ("object_visibility_future", "object_visibility_future", "object_visibility_future_mask"),
    ("next_best_view_score", "next_best_view_score", "next_best_view_score_mask"),
]

# Short display names so "label + indicator" fits side-by-side in a 256px-wide cell.
DISPLAY_NAMES: dict[str, str] = {
    "task_progress": "progress",
    "subgoal_feasibility": "subgoal",
    "manipulation_readiness": "readiness",
    "failure_risk": "failure_risk",
    "object_visibility_future": "visibility",
    "next_best_view_score": "best_view",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a two-camera video overlay of MoWA future-label sidecars."
    )
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
        help="Output video path. Default: docs_zh/mowa/head_labels_check/<task>_episode_XXXXXX_video.mp4",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Output video FPS. Default: 20",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=64,
        help="Height of the indicator panel in pixels. Default: 64 (~20% of 256px camera height)",
    )
    parser.add_argument(
        "--curve-dpi",
        type=int,
        default=300,
        help="DPI for the high-resolution curve image. Default: 300",
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
        candidate = (
            date_dir
            / "lerobot"
            / "mowa_future_labels"
            / task
            / f"episode_{episode_index:06d}.parquet"
        )
        if candidate.is_file():
            return candidate
    return None


def find_video_path(data_root: Path, task: str, episode_index: int, camera: str) -> Path | None:
    """Find the camera video MP4 for a given task/episode."""
    atomic_root = data_root / "v1.0" / "target" / "atomic" / task
    if not atomic_root.is_dir():
        return None
    for date_dir in sorted(atomic_root.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.isdigit():
            continue
        candidate = (
            date_dir
            / "lerobot"
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera}"
            / f"episode_{episode_index:06d}.mp4"
        )
        if candidate.is_file():
            return candidate
    return None


def _color_for_value(value: float, masked: bool) -> tuple[int, int, int]:
    """Return BGR color for an indicator dot."""
    if masked:
        return (128, 128, 128)  # gray
    if value >= 0.7:
        return (0, 255, 0)  # bright green
    if value >= 0.3:
        return (0, 255, 255)  # yellow
    return (0, 0, 255)  # red


def _render_indicator_panel(
    data: dict[str, list],
    frame_index: np.ndarray,
    timestep: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Render the 2x3 bright-dot indicator grid for a single frame.

    Layout per row: [label] [dot]   [label] [dot]
    """
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    n_cols = 2
    n_rows = 3
    cell_w = width // n_cols
    cell_h = height // n_rows

    # Compact panel: scale font and dot to the small cell size.
    max_font_size = max(8, min(14, cell_h // 3))
    margin = max(4, cell_w // 32)
    radius = max(5, min(cell_w, cell_h) // 3 - 1)
    max_text_width = cell_w - 2 * margin - 2 * radius - 4

    for idx, (title, value_key, mask_key) in enumerate(LABEL_SPECS):
        col = idx % n_cols
        row = idx // n_cols
        x0 = col * cell_w
        y0 = row * cell_h

        values = np.asarray(data[value_key], dtype=np.float64)
        masked = np.asarray(data[mask_key], dtype=bool) if mask_key else np.zeros(len(values), dtype=bool)

        current_value = float(values[timestep])
        current_masked = bool(masked[timestep])
        color = _color_for_value(current_value, current_masked)

        display_name = DISPLAY_NAMES.get(title, title)

        # Pick a font size that lets the label fit without touching the dot.
        font_size = max_font_size
        font_label = None
        while font_size >= 6:
            try:
                candidate = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
                )
            except Exception:
                candidate = ImageFont.load_default()
                break
            bbox = draw.textbbox((0, 0), display_name, font=candidate)
            if bbox[2] - bbox[0] <= max_text_width:
                font_label = candidate
                break
            font_size -= 1
        if font_label is None:
            font_label = ImageFont.load_default()

        # Label on the left, vertically centered.
        label_bbox = draw.textbbox((0, 0), display_name, font=font_label)
        label_h = label_bbox[3] - label_bbox[1]
        text_x = x0 + margin
        text_y = y0 + (cell_h - label_h) // 2
        draw.text((text_x, text_y), display_name, fill=(220, 220, 220), font=font_label)

        # Bright-dot indicator on the right, vertically centered.
        dot_cx = x0 + cell_w - margin - radius
        dot_cy = y0 + cell_h // 2
        draw.ellipse(
            [(dot_cx - radius, dot_cy - radius), (dot_cx + radius, dot_cy + radius)],
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )

    return np.array(img)


def _render_hd_curve_image(
    data: dict[str, list],
    frame_index: np.ndarray,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Render a high-resolution static curve plot for all six heads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(LABEL_SPECS), 1, figsize=(12, 14), sharex=True)
    if len(LABEL_SPECS) == 1:
        axes = [axes]

    for ax, (title, value_key, mask_key) in zip(axes, LABEL_SPECS, strict=True):
        values = np.asarray(data[value_key], dtype=np.float64)
        ax.plot(frame_index, values, label=title, linewidth=1.0, color="tab:blue")

        if mask_key:
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

        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("frame_index", fontsize=9)
    fig.suptitle(f"Future head labels\n{output_path.stem.replace('_labels', '')}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sidecar_path = find_sidecar_path(args.data_root, args.task, args.episode)
    if sidecar_path is None:
        raise FileNotFoundError(
            f"No sidecar found for {args.task} episode {args.episode} under {args.data_root}"
        )

    left_path = find_video_path(args.data_root, args.task, args.episode, LEFT_CAMERA)
    wrist_path = find_video_path(args.data_root, args.task, args.episode, WRIST_CAMERA)
    if left_path is None:
        raise FileNotFoundError(
            f"No {LEFT_CAMERA} video found for {args.task} episode {args.episode}"
        )
    if wrist_path is None:
        raise FileNotFoundError(
            f"No {WRIST_CAMERA} video found for {args.task} episode {args.episode}"
        )

    table = pq.read_table(sidecar_path)
    data = table.to_pydict()
    frame_index = np.asarray(
        data.get("frame_index", np.arange(len(data["task_progress"]))), dtype=np.int64
    )

    left_cap = cv2.VideoCapture(str(left_path))
    wrist_cap = cv2.VideoCapture(str(wrist_path))
    if not left_cap.isOpened():
        raise RuntimeError(f"Could not open video {left_path}")
    if not wrist_cap.isOpened():
        raise RuntimeError(f"Could not open video {wrist_path}")

    video_width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = left_cap.get(cv2.CAP_PROP_FPS) or args.fps
    left_frames = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wrist_frames = int(wrist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_frames = len(frame_index)
    if left_frames != expected_frames or wrist_frames != expected_frames:
        print(
            f"Warning: left={left_frames}, wrist={wrist_frames}, sidecar={expected_frames}; "
            f"using min of the three."
        )
    n_frames = min(left_frames, wrist_frames, expected_frames)

    if args.output is None:
        out_dir = Path("docs_zh/mowa/head_labels_check")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / f"{args.task}_episode_{args.episode:06d}_video.mp4"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    curve_image_path = args.output.with_name(args.output.stem + "_labels.png")
    _render_hd_curve_image(data, frame_index, curve_image_path, dpi=args.curve_dpi)

    output_width = video_width * 2
    output_height = args.panel_height + video_height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(args.output),
        fourcc,
        float(video_fps),
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    for t in range(n_frames):
        ret_left, left_frame = left_cap.read()
        ret_wrist, wrist_frame = wrist_cap.read()
        if not ret_left or not ret_wrist:
            break

        panel = _render_indicator_panel(
            data=data,
            frame_index=frame_index,
            timestep=t,
            width=output_width,
            height=args.panel_height,
        )
        video_row = np.hstack([left_frame, wrist_frame])
        composite = np.vstack([panel, video_row])
        writer.write(composite)

    left_cap.release()
    wrist_cap.release()
    writer.release()
    print(f"Wrote video to {args.output} ({n_frames} frames @ {video_fps:.1f} fps)")
    print(f"Wrote HD curve image to {curve_image_path}")


if __name__ == "__main__":
    main()
