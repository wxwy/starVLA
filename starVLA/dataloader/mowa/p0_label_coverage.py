"""MoWA P0 label coverage smoke utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starVLA.mowa_constants import MOWA_P0_FULL_HEADS
from starVLA.dataloader.mowa.schema import DATA_GATE, TBD


@dataclass(frozen=True)
class MoWAP0HeadCoverage:
    head: str
    status: str
    source_fields: tuple[str, ...]
    mask_rule: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "status": self.status,
            "source_fields": self.source_fields,
            "mask_rule": self.mask_rule,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MoWAP0LabelCoverageReport:
    dataset_path: str
    sampled_episode_indices: tuple[int, ...]
    available_columns: tuple[str, ...]
    head_coverage: tuple[MoWAP0HeadCoverage, ...]
    constructible_heads: tuple[str, ...]
    masked_heads: tuple[str, ...]
    unresolved_items: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "sampled_episode_indices": self.sampled_episode_indices,
            "available_columns": self.available_columns,
            "head_coverage": [coverage.to_dict() for coverage in self.head_coverage],
            "constructible_heads": self.constructible_heads,
            "masked_heads": self.masked_heads,
            "unresolved_items": list(self.unresolved_items),
        }


def inspect_mowa_p0_label_coverage(
    dataset_path: Path | str,
    episode_indices: tuple[int, ...] = (0, 1, 4),
) -> MoWAP0LabelCoverageReport:
    """只读检查 P0 七类 head 的字段可构造性。

    该函数不生成训练标签，不定义阈值，不启动 P0 模型。
    """

    root = Path(dataset_path)
    columns_by_episode = tuple(
        _read_parquet_columns(root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet")
        for episode_index in episode_indices
    )
    available_columns = tuple(sorted(set().union(*columns_by_episode))) if columns_by_episode else ()
    coverage = tuple(_build_head_coverage(head, available_columns) for head in MOWA_P0_FULL_HEADS)
    constructible_heads = tuple(item.head for item in coverage if item.status == "candidate_constructible")
    masked_heads = tuple(item.head for item in coverage if item.status != "candidate_constructible")

    return MoWAP0LabelCoverageReport(
        dataset_path=str(root),
        sampled_episode_indices=episode_indices,
        available_columns=available_columns,
        head_coverage=coverage,
        constructible_heads=constructible_heads,
        masked_heads=masked_heads,
        unresolved_items=(
            "Coverage is field-level only; label builders and thresholds remain Data Gate.",
            "FullHeads can proceed only with masks for non-constructible heads.",
            "No future action label may enter WAM inputs.",
        ),
    )


def _read_parquet_columns(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA P0 label coverage smoke requires pyarrow.") from exc
    return tuple(pq.ParquetFile(path).schema_arrow.names)


def _build_head_coverage(head: str, available_columns: tuple[str, ...]) -> MoWAP0HeadCoverage:
    columns = set(available_columns)
    if head == "task_progress":
        required = ("frame_index", "timestamp", "episode_index")
        return _coverage(
            head,
            required,
            columns,
            "mask if frame_index/timestamp/episode length unavailable",
            "Candidate from frame_index normalized by episode length; threshold remains Data Gate.",
        )
    if head == "manipulation_readiness":
        required = ("observation.state", "action")
        return _coverage(
            head,
            required,
            columns,
            "mask until readiness proxy is validated",
            "State/action fields exist, but readiness definition remains Data Gate.",
            force_data_gate=True,
        )
    if head == "failure_risk":
        return MoWAP0HeadCoverage(
            head=head,
            status="masked",
            source_fields=("failure_annotation",),
            mask_rule="mask by default",
            notes="No explicit failure annotation in current Lerobot schema.",
        )
    if head == "next_best_view_score":
        return MoWAP0HeadCoverage(
            head=head,
            status="masked",
            source_fields=("view_score", "visibility_label"),
            mask_rule="mask by default",
            notes="Requires a view/visibility proxy definition; not available from scalar parquet fields.",
        )
    if head == "subgoal_feasibility":
        required = ("next.reward", "next.done", "frame_index")
        return _coverage(
            head,
            required,
            columns,
            "mask until feasibility proxy is validated",
            "Reward/done/progress fields exist, but feasibility target remains Data Gate.",
            force_data_gate=True,
        )
    if head == "object_visibility_future":
        return MoWAP0HeadCoverage(
            head=head,
            status="masked",
            source_fields=("future_video", "object_visibility_proxy"),
            mask_rule="mask by default",
            notes="Requires video decode or a validated visibility proxy.",
        )
    if head == "action_outcome_class":
        required = ("next.reward", "next.done")
        return _coverage(
            head,
            required,
            columns,
            "mask if reward/done unavailable",
            "Candidate from next.reward / next.done; class mapping is frozen for E-001 initial target.",
        )

    return MoWAP0HeadCoverage(
        head=head,
        status=TBD,
        source_fields=(),
        mask_rule="mask unknown head",
        notes="Unknown P0 head.",
    )


def _coverage(
    head: str,
    required: tuple[str, ...],
    columns: set[str],
    mask_rule: str,
    notes: str,
    force_data_gate: bool = False,
) -> MoWAP0HeadCoverage:
    missing = tuple(field for field in required if field not in columns)
    status = "candidate_constructible" if not missing and not force_data_gate else DATA_GATE
    if missing:
        status = "masked"
        notes = f"Missing fields: {missing}."
    return MoWAP0HeadCoverage(
        head=head,
        status=status,
        source_fields=required,
        mask_rule=mask_rule,
        notes=notes,
    )
