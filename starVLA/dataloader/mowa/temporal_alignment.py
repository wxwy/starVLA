"""原始 RoboCasa episode 的多视角时序对齐校验。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MOWA_VIDEO_KEY = "observation.images.robot0_agentview_left"


@dataclass(frozen=True)
class MoWATemporalAlignmentEpisodeReport:
    """单个 episode 的 parquet/video 帧数校验结果。"""

    episode_index: int
    parquet_row_count: int | None
    video_frame_counts: dict[str, int | None]
    errors: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "parquet_row_count": self.parquet_row_count,
            "video_frame_counts": self.video_frame_counts,
            "errors": list(self.errors),
            "is_valid": self.is_valid,
        }


@dataclass(frozen=True)
class MoWATemporalAlignmentReport:
    """一个 LeRobot 数据目录的原始时序对齐校验结果。"""

    dataset_path: str
    video_keys: tuple[str, ...]
    checked_episode_count: int
    valid_episode_count: int
    invalid_episode_count: int
    episodes: tuple[MoWATemporalAlignmentEpisodeReport, ...]

    @property
    def go_no_go(self) -> str:
        if self.checked_episode_count == 0:
            return "No-Go: no episode parquet files found"
        if self.invalid_episode_count:
            return "No-Go: video/state/action temporal alignment failed"
        return "TBD: video/state/action temporal alignment passed"

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_path": self.dataset_path,
            "video_keys": list(self.video_keys),
            "checked_episode_count": self.checked_episode_count,
            "valid_episode_count": self.valid_episode_count,
            "invalid_episode_count": self.invalid_episode_count,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "go_no_go": self.go_no_go,
        }


def validate_mowa_episode_temporal_alignment(
    dataset_path: Path | str,
    *,
    video_keys: tuple[str, ...] = (DEFAULT_MOWA_VIDEO_KEY,),
    episode_indices: tuple[int, ...] | None = None,
    video_frame_counter: Callable[[Path], int] | None = None,
) -> MoWATemporalAlignmentReport:
    """校验每个指定视角视频是否与 parquet 的 state/action 时间轴严格对齐。

    ``video_frame_counter`` 用于复用其他视频后端或在调用方测试时注入计数实现。
    未传入时使用 OpenCV 的 MP4 帧数元信息。
    """

    dataset_path = Path(dataset_path)
    if not video_keys:
        raise ValueError("video_keys must be non-empty.")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    selected_indices = set(episode_indices) if episode_indices is not None else None
    parquet_paths = sorted((dataset_path / "data" / "chunk-000").glob("episode_*.parquet"))
    if selected_indices is not None:
        parquet_paths = [
            path for path in parquet_paths
            if _episode_index_from_parquet_path(path) in selected_indices
        ]

    frame_counter = video_frame_counter or count_video_frames_opencv
    episodes = tuple(
        _validate_episode(dataset_path, parquet_path, video_keys, frame_counter)
        for parquet_path in parquet_paths
    )
    invalid_episode_count = sum(not episode.is_valid for episode in episodes)
    return MoWATemporalAlignmentReport(
        dataset_path=str(dataset_path),
        video_keys=video_keys,
        checked_episode_count=len(episodes),
        valid_episode_count=len(episodes) - invalid_episode_count,
        invalid_episode_count=invalid_episode_count,
        episodes=episodes,
    )


def count_video_frames_opencv(video_path: Path) -> int:
    """返回视频容器声明的帧数，无法读取时抛出可定位异常。"""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Temporal alignment validation requires opencv-python.") from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count < 0:
        raise ValueError(f"Invalid video frame count {frame_count}: {video_path}")
    return frame_count


def _validate_episode(
    dataset_path: Path,
    parquet_path: Path,
    video_keys: tuple[str, ...],
    video_frame_counter: Callable[[Path], int],
) -> MoWATemporalAlignmentEpisodeReport:
    episode_index = _episode_index_from_parquet_path(parquet_path)
    errors: list[str] = []
    video_frame_counts: dict[str, int | None] = {}
    parquet_row_count: int | None = None

    try:
        parquet_row_count = _read_state_action_row_count(parquet_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    for video_key in video_keys:
        video_path = _resolve_video_path(dataset_path, video_key, episode_index)
        if not video_path.is_file():
            video_frame_counts[video_key] = None
            errors.append(f"missing video for {video_key}: {video_path}")
            continue
        try:
            frame_count = int(video_frame_counter(video_path))
            video_frame_counts[video_key] = frame_count
        except Exception as exc:  # noqa: BLE001
            video_frame_counts[video_key] = None
            errors.append(f"failed to count {video_key}: {exc}")
            continue
        if parquet_row_count is not None and frame_count != parquet_row_count:
            errors.append(
                f"frame count mismatch for {video_key}: "
                f"video={frame_count}, state/action={parquet_row_count}"
            )

    return MoWATemporalAlignmentEpisodeReport(
        episode_index=episode_index,
        parquet_row_count=parquet_row_count,
        video_frame_counts=video_frame_counts,
        errors=tuple(errors),
    )


def _read_state_action_row_count(parquet_path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Temporal alignment validation requires pyarrow.") from exc

    parquet = pq.ParquetFile(parquet_path)
    column_names = set(parquet.schema_arrow.names)
    missing_columns = {"observation.state", "action"} - column_names
    if missing_columns:
        raise ValueError(f"Missing required parquet columns {sorted(missing_columns)}: {parquet_path}")
    return int(parquet.metadata.num_rows)


def _episode_index_from_parquet_path(parquet_path: Path) -> int:
    try:
        return int(parquet_path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unable to parse episode index from {parquet_path}") from exc


def _resolve_video_path(dataset_path: Path, video_key: str, episode_index: int) -> Path:
    return dataset_path / "videos" / "chunk-000" / video_key / f"episode_{episode_index:06d}.mp4"
