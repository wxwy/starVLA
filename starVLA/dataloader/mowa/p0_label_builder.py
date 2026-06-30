"""MoWA P0 constructible label builder smoke utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starVLA.dataloader.mowa.robocasa365_adapter import MOWA_P0_HEADS
from starVLA.dataloader.mowa.schema import DATA_GATE


MOWA_P0_CONSTRUCTIBLE_HEADS = ("task_progress", "action_outcome_class")


@dataclass(frozen=True)
class MoWAP0LabelSmokeSample:
    episode_index: int
    row_index: int
    labels: dict[str, Any]
    masks: dict[str, bool]
    sources: dict[str, tuple[str, ...]]
    data_gate: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "row_index": self.row_index,
            "labels": self.labels,
            "masks": self.masks,
            "sources": self.sources,
            "data_gate": self.data_gate,
        }


@dataclass(frozen=True)
class MoWAP0ConstructibleLabelSmoke:
    dataset_path: str
    sampled_episode_indices: tuple[int, ...]
    constructible_heads: tuple[str, ...]
    masked_heads: tuple[str, ...]
    sample_count: int
    samples: tuple[MoWAP0LabelSmokeSample, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "sampled_episode_indices": self.sampled_episode_indices,
            "constructible_heads": self.constructible_heads,
            "masked_heads": self.masked_heads,
            "sample_count": self.sample_count,
            "samples": [sample.to_dict() for sample in self.samples],
            "notes": list(self.notes),
        }


def build_mowa_p0_constructible_label_smoke(
    dataset_path: Path | str,
    episode_indices: tuple[int, ...] = (0, 1, 4),
    preview_rows: int = 8,
) -> MoWAP0ConstructibleLabelSmoke:
    """Build smoke-only P0 labels for currently constructible heads.

    该函数只读取 parquet 标量字段，输出 dry-run targets/masks。
    它不定义 production 阈值，不冻结 action outcome class mapping，
    不启动训练，也不把任何标签放入 WAM inputs。
    """

    root = Path(dataset_path)
    samples: list[MoWAP0LabelSmokeSample] = []
    for episode_index in episode_indices:
        parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        samples.extend(_build_episode_samples(parquet_path, episode_index, preview_rows))

    masked_heads = tuple(head for head in MOWA_P0_HEADS if head not in MOWA_P0_CONSTRUCTIBLE_HEADS)
    return MoWAP0ConstructibleLabelSmoke(
        dataset_path=str(root),
        sampled_episode_indices=episode_indices,
        constructible_heads=MOWA_P0_CONSTRUCTIBLE_HEADS,
        masked_heads=masked_heads,
        sample_count=len(samples),
        samples=tuple(samples),
        notes=(
            "Smoke-only labels are targets, not WAM inputs.",
            "task_progress is normalized by episode row count for dry-run only.",
            "action_outcome_class keeps raw next.reward/next.done; class mapping remains Data Gate.",
            "Non-constructible P0 heads stay masked.",
        ),
    )


def build_mowa_p0_label_smoke_sample(
    dataset_path: Path | str,
    episode_index: int,
    row_index: int,
) -> MoWAP0LabelSmokeSample:
    """Build one dry-run label sample for a concrete episode row."""

    root = Path(dataset_path)
    parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    samples = _build_episode_samples(
        parquet_path,
        episode_index=episode_index,
        preview_rows=row_index + 1,
    )
    if row_index >= len(samples):
        raise IndexError(
            f"MoWA P0 label row_index out of range: {row_index}, available={len(samples)}."
        )
    return samples[row_index]


def _build_episode_samples(
    parquet_path: Path,
    episode_index: int,
    preview_rows: int,
) -> list[MoWAP0LabelSmokeSample]:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"MoWA P0 label builder parquet not found: {parquet_path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA P0 label builder smoke requires pyarrow.") from exc

    parquet_file = pq.ParquetFile(parquet_path)
    required_columns = ("frame_index", "next.reward", "next.done")
    columns = set(parquet_file.schema_arrow.names)
    missing = tuple(column for column in required_columns if column not in columns)
    if missing:
        raise ValueError(f"MoWA P0 label builder missing required columns: {missing}")

    row_count = parquet_file.metadata.num_rows
    table = parquet_file.read(columns=list(required_columns)).slice(0, preview_rows).to_pydict()
    frame_indices = table["frame_index"]
    rewards = table["next.reward"]
    dones = table["next.done"]
    denominator = max(row_count - 1, 1)

    samples = []
    for row_index, (frame_index, reward, done) in enumerate(zip(frame_indices, rewards, dones)):
        masks = {head: head in MOWA_P0_CONSTRUCTIBLE_HEADS for head in MOWA_P0_HEADS}
        samples.append(
            MoWAP0LabelSmokeSample(
                episode_index=episode_index,
                row_index=row_index,
                labels={
                    "task_progress": float(frame_index) / denominator,
                    "action_outcome_class": {
                        "next_reward": float(reward),
                        "next_done": bool(done),
                        "class_mapping_status": DATA_GATE,
                    },
                },
                masks=masks,
                sources={
                    "task_progress": ("frame_index", "episode_row_count"),
                    "action_outcome_class": ("next.reward", "next.done"),
                },
                data_gate={
                    "thresholds": DATA_GATE,
                    "class_mapping": DATA_GATE,
                    "window": DATA_GATE,
                },
            )
        )
    return samples
