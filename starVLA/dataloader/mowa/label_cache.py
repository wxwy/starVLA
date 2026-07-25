"""Generic sidecar cache writer/loader for MoWA atomic-task future labels.

This module is task-agnostic: it delegates the per-episode computation to an
``AtomicTaskLabelBuilder`` and handles the common parquet I/O and directory
layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.mowa.atomic_task_label_builder import AtomicTaskLabelBuilder

# Standard sidecar columns.  Task-specific builders may append extra debug
# columns, but these must always be present.
LABEL_CACHE_STANDARD_COLUMNS = (
    "frame_index",
    "failure_risk",
    "failure_risk_mask",
    "subgoal_feasibility",
    "subgoal_feasibility_mask",
    "manipulation_readiness",
    "manipulation_readiness_mask",
    "object_visibility_future",
    "object_visibility_future_mask",
    "next_best_view_score",
    "next_best_view_score_mask",
)


def build_label_cache_for_episode(
    builder: AtomicTaskLabelBuilder,
    parquet_path: Path,
    states_path: Path,
    model_path: Path | None = None,
    ep_meta_path: Path | None = None,
    *,
    failure_risk_horizon: int = 10,
    subgoal_horizon: int = 20,
    readiness_horizon: int | None = None,
    readiness_progress_delta: float | None = None,
    readiness_distance_threshold: float | None = None,
    enable_kinematics: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Compute labels for one episode using ``builder``.

    This is a thin wrapper around ``builder.build_cache_for_episode`` that
    validates the returned dict contains the standard sidecar columns.
    """
    cache = builder.build_cache_for_episode(
        parquet_path=parquet_path,
        states_path=states_path,
        model_path=model_path,
        ep_meta_path=ep_meta_path,
        failure_risk_horizon=failure_risk_horizon,
        subgoal_horizon=subgoal_horizon,
        readiness_horizon=readiness_horizon,
        readiness_progress_delta=readiness_progress_delta,
        readiness_distance_threshold=readiness_distance_threshold,
        enable_kinematics=enable_kinematics,
        repo_root=repo_root,
    )
    missing = tuple(column for column in LABEL_CACHE_STANDARD_COLUMNS if column not in cache)
    if missing:
        raise ValueError(
            f"Task {builder.task_name} builder missing standard sidecar columns: {missing}"
        )
    return cache


def _write_cache_parquet(
    cache: dict[str, Any],
    output_dir: Path,
    episode_index: int,
    pa: Any,
    pq: Any,
) -> None:
    """Write a single episode sidecar parquet from a builder cache dict."""
    columns: dict[str, Any] = {
        "frame_index": cache["frame_index"],
        "failure_risk": cache["failure_risk"],
        "failure_risk_mask": cache["failure_risk_mask"],
        "subgoal_feasibility": cache["subgoal_feasibility"],
        "subgoal_feasibility_mask": cache["subgoal_feasibility_mask"],
        "manipulation_readiness": cache["manipulation_readiness"],
        "manipulation_readiness_mask": cache["manipulation_readiness_mask"],
        "object_visibility_future": cache["object_visibility_future"],
        "object_visibility_future_mask": cache["object_visibility_future_mask"],
        "next_best_view_score": cache["next_best_view_score"],
        "next_best_view_score_mask": cache["next_best_view_score_mask"],
    }
    # Append optional task-specific debug columns (1-D arrays only).
    for key, value in cache.items():
        if key not in columns and isinstance(value, np.ndarray) and value.ndim == 1:
            columns[key] = value

    table = pa.table(columns)
    schema_version = str(cache.get("schema_version", "unknown"))
    table = table.replace_schema_metadata({"schema_version": schema_version})
    output_path = output_dir / f"episode_{episode_index:06d}.parquet"
    pq.write_table(table, output_path)


def write_label_cache(
    builder: AtomicTaskLabelBuilder,
    dataset_path: Path | str,
    output_dir: Path | str | None = None,
    *,
    failure_risk_horizon: int = 10,
    subgoal_horizon: int = 20,
    readiness_horizon: int | None = None,
    readiness_progress_delta: float | None = None,
    readiness_distance_threshold: float | None = None,
    enable_kinematics: bool = True,
    max_episodes: int | None = None,
    repo_root: Path | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Pre-compute future labels for all episodes of a task and write sidecars.

    Sidecars are written to ``<dataset_path>/mowa_future_labels/<task_name>/``
    unless ``output_dir`` is provided.

    If ``skip_existing`` is True, episodes whose sidecar parquet already exists
    are read back and aggregated instead of recomputed.  This is useful for
    regenerating a manifest after a worker pool is interrupted.
    """
    dataset_path = Path(dataset_path)
    if output_dir is None:
        output_dir = dataset_path / "mowa_future_labels" / builder.sidecar_dir_name()
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Label cache requires pyarrow.") from exc

    data_dir = dataset_path / "data" / "chunk-000"
    parquet_paths = sorted(data_dir.glob("episode_*.parquet"))
    if max_episodes is not None:
        parquet_paths = parquet_paths[:max_episodes]

    processed = 0
    skipped_no_extras = 0
    skipped_empty = 0
    total_rows = 0
    subgoal_positive = 0
    readiness_positive = 0
    failure_risk_labeled = 0
    object_visibility_positive = 0
    next_best_view_sum = 0.0
    schema_version = "unknown"

    for parquet_path in parquet_paths:
        episode_index = int(parquet_path.stem.split("_")[-1])
        extras_dir = dataset_path / "extras" / f"episode_{episode_index:06d}"
        states_path = extras_dir / "states.npz"
        if not states_path.is_file():
            skipped_no_extras += 1
            continue

        output_path = output_dir / f"episode_{episode_index:06d}.parquet"
        if skip_existing and output_path.is_file():
            table = pq.read_table(output_path)
            data = table.to_pydict()
            row_count = len(data["frame_index"])
            if row_count == 0:
                skipped_empty += 1
                continue
            processed += 1
            total_rows += row_count
            subgoal_positive += int(np.asarray(data["subgoal_feasibility"], dtype=np.float64).sum())
            readiness_positive += int(np.asarray(data["manipulation_readiness"], dtype=np.float64).sum())
            failure_risk_labeled += int((~np.asarray(data["failure_risk_mask"], dtype=bool)).sum())
            object_visibility_positive += int(np.asarray(data["object_visibility_future"], dtype=np.float64).sum())
            ovf_mask = ~np.asarray(data["object_visibility_future_mask"], dtype=bool)
            if ovf_mask.any():
                nbv_active = np.asarray(data["next_best_view_score"], dtype=np.float64)[ovf_mask]
                next_best_view_sum += float(nbv_active.sum())
            schema_version = (table.schema.metadata or {}).get(b"schema_version", b"unknown").decode()
            continue



        cache = build_label_cache_for_episode(
            builder=builder,
            parquet_path=parquet_path,
            states_path=states_path,
            model_path=extras_dir / "model.xml.gz",
            ep_meta_path=extras_dir / "ep_meta.json",
            failure_risk_horizon=failure_risk_horizon,
            subgoal_horizon=subgoal_horizon,
            readiness_horizon=readiness_horizon,
            readiness_progress_delta=readiness_progress_delta,
            readiness_distance_threshold=readiness_distance_threshold,
            enable_kinematics=enable_kinematics,
            repo_root=repo_root,
        )
        if cache["row_count"] == 0:
            skipped_empty += 1
            continue

        _write_cache_parquet(cache, output_dir, episode_index, pa, pq)
        schema_version = str(cache.get("schema_version", "unknown"))

        processed += 1
        total_rows += int(cache["row_count"])
        subgoal_positive += int(np.asarray(cache["subgoal_feasibility"]).sum())
        readiness_positive += int(np.asarray(cache["manipulation_readiness"]).sum())
        failure_risk_labeled += int((~np.asarray(cache["failure_risk_mask"], dtype=bool)).sum())
        object_visibility_positive += int(np.asarray(cache["object_visibility_future"]).sum())
        ovf_mask = ~np.asarray(cache["object_visibility_future_mask"], dtype=bool)
        if ovf_mask.any():
            nbv_active = np.asarray(cache["next_best_view_score"])[ovf_mask]
            next_best_view_sum += float(nbv_active.sum())

    return {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "task_name": builder.task_name,
        "failure_risk_horizon": failure_risk_horizon,
        "subgoal_horizon": subgoal_horizon,
        "readiness_horizon": readiness_horizon,
        "readiness_progress_delta": readiness_progress_delta,
        "readiness_distance_threshold": readiness_distance_threshold,
        "enable_kinematics": enable_kinematics,
        "parquet_count": len(parquet_paths),
        "processed_episode_count": processed,
        "skipped_no_extras_count": skipped_no_extras,
        "skipped_empty_count": skipped_empty,
        "total_row_count": total_rows,
        "subgoal_positive_count": subgoal_positive,
        "readiness_positive_count": readiness_positive,
        "failure_risk_labeled_count": failure_risk_labeled,
        "object_visibility_positive_count": object_visibility_positive,
        "next_best_view_score_sum": next_best_view_sum,
        "schema_version": schema_version,
        "go_no_go": (
            f"TBD: {builder.task_name} label cache built; review distribution before unmasking"
            if processed > 0
            else f"No-Go: no {builder.task_name} episodes with extras/states.npz"
        ),
    }


def load_label_cache_for_episode(
    builder: AtomicTaskLabelBuilder,
    dataset_path: Path | str,
    episode_index: int,
) -> dict[str, Any] | None:
    """Load a pre-computed label sidecar for one episode.

    Returns a dict keyed like ``mowa_future_targets`` / ``mowa_future_masks``,
    or ``None`` if no sidecar exists.
    """
    dataset_path = Path(dataset_path)
    cache_dir = dataset_path / "mowa_future_labels" / builder.sidecar_dir_name()
    cache_path = cache_dir / f"episode_{episode_index:06d}.parquet"
    if not cache_path.is_file():
        return None

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Label cache requires pyarrow.") from exc

    table = pq.read_table(cache_path)
    data = table.to_pydict()
    has_kinematics = "eef_to_handle_distance" in data

    def _to_array(values: list[Any], dtype: Any) -> np.ndarray:
        if len(values) == 1:
            return np.asarray([float(values[0])], dtype=dtype)
        return np.asarray(values, dtype=dtype)

    failure_risk_values = data["failure_risk"]
    return {
        "frame_index": np.asarray(data["frame_index"], dtype=np.int64),
        "has_kinematics": has_kinematics,
        "labels": {
            "failure_risk": _to_array(failure_risk_values, np.float64),
            "subgoal_feasibility": np.asarray(data["subgoal_feasibility"], dtype=np.float64),
            "manipulation_readiness": np.asarray(data["manipulation_readiness"], dtype=np.float64),
            "object_visibility_future": np.asarray(data["object_visibility_future"], dtype=np.float64),
            "next_best_view_score": np.asarray(data["next_best_view_score"], dtype=np.float64),
        },
        "masks": {
            "failure_risk": ~np.asarray(data["failure_risk_mask"], dtype=bool),
            "subgoal_feasibility": ~np.asarray(data["subgoal_feasibility_mask"], dtype=bool),
            "manipulation_readiness": ~np.asarray(data["manipulation_readiness_mask"], dtype=bool),
            "object_visibility_future": ~np.asarray(data["object_visibility_future_mask"], dtype=bool),
            "next_best_view_score": ~np.asarray(data["next_best_view_score_mask"], dtype=bool),
        },
    }


def label_cache_available(
    builder: AtomicTaskLabelBuilder,
    dataset_path: Path | str,
) -> bool:
    """Return True if at least one episode sidecar exists for the dataset."""
    dataset_path = Path(dataset_path)
    cache_dir = dataset_path / "mowa_future_labels" / builder.sidecar_dir_name()
    if not cache_dir.is_dir():
        return False
    return any(cache_dir.glob("episode_*.parquet"))
