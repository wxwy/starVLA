"""StarFlow-VLA LIBERO eval report helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPORT_FILENAME = "eval_report.json"


def _sha256_file(path: Path, hasher: hashlib._Hash) -> None:
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)


def sha256_path(path: str | Path) -> str:
    """Return a deterministic hash over a file or directory tree.

    For checkpoint directories we fingerprint large weight shards by relative path
    and file size, while hashing the full contents of lightweight metadata files.
    This keeps eval startup fast enough for smoke runs.
    """
    path = Path(path)
    hasher = hashlib.sha256()
    if path.is_file():
        hasher.update(path.name.encode("utf-8"))
        _sha256_file(path, hasher)
        return hasher.hexdigest()

    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        hasher.update(str(child.relative_to(path)).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        if stat.st_size <= 1024 * 1024:
            _sha256_file(child, hasher)
    return hasher.hexdigest()


def _candidate_search_dirs(pretrained_path: Path) -> list[Path]:
    candidates: list[Path] = []
    if pretrained_path.is_dir():
        candidates.extend([pretrained_path, pretrained_path.parent, pretrained_path.parent.parent])
    else:
        candidates.extend([pretrained_path.parent, pretrained_path.parent.parent])
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_config_scalar(config_text: str, key: str) -> str | None:
    pattern = rf"^\s*{re.escape(key)}:\s*['\"]?([^'\"\n]+)"
    match = re.search(pattern, config_text, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def load_eval_metadata(pretrained_path: str | Path) -> dict[str, Any]:
    """Load checkpoint/config/data metadata needed by the eval report."""
    checkpoint_path = Path(pretrained_path)
    search_dirs = _candidate_search_dirs(checkpoint_path)

    if checkpoint_path.is_file():
        mapping_candidates = [
            checkpoint_path.with_name(f"{checkpoint_path.name}.starflow_mapping.json"),
            checkpoint_path.parent / "starflow_mapping.json",
        ]
    else:
        mapping_candidates = [checkpoint_path / "starflow_mapping.json"]

    mapping_path = _find_first_existing(mapping_candidates)
    config_path = _find_first_existing([directory / "config.yaml" for directory in search_dirs])
    dataset_statistics_path = _find_first_existing([directory / "dataset_statistics.json" for directory in search_dirs])

    config_text = config_path.read_text(encoding="utf-8") if config_path is not None else ""
    config_hash = sha256_path(config_path) if config_path is not None else None
    checkpoint_hash = sha256_path(checkpoint_path) if checkpoint_path.exists() else None
    dataset_statistics_hash = sha256_path(dataset_statistics_path) if dataset_statistics_path is not None else None

    starflow_mapping = None
    if mapping_path is not None:
        starflow_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    data_version = {
        "data_mix": _extract_config_scalar(config_text, "data_mix"),
        "data_root_dir": _extract_config_scalar(config_text, "data_root_dir"),
        "config_schema": _extract_config_scalar(config_text, "version_id"),
        "dataset_statistics_hash": dataset_statistics_hash,
    }

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "config_path": str(config_path) if config_path is not None else None,
        "config_hash": config_hash,
        "dataset_statistics_path": str(dataset_statistics_path) if dataset_statistics_path is not None else None,
        "data_version": data_version,
        "starflow_mapping": starflow_mapping,
    }


def infer_failure_category(*, success: bool, runtime_error: str | None) -> str | None:
    if success:
        return None
    if runtime_error:
        return "runtime_error"
    return "timeout_no_success"


def build_eval_report(
    *,
    args: dict[str, Any],
    metadata: dict[str, Any],
    total_episodes: int,
    total_successes: int,
    episode_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-serializable eval report."""
    failure_counter = Counter()
    task_rollup: dict[str, dict[str, Any]] = defaultdict(lambda: {"episodes": 0, "successes": 0, "failure_category": Counter()})

    for record in episode_records:
        task_entry = task_rollup[record["task_description"]]
        task_entry["episodes"] += 1
        task_entry["successes"] += int(record["success"])
        failure_category = record.get("failure_category")
        if failure_category:
            failure_counter[failure_category] += 1
            task_entry["failure_category"][failure_category] += 1

    tasks = []
    for task_description, task_entry in sorted(task_rollup.items()):
        episodes = int(task_entry["episodes"])
        successes = int(task_entry["successes"])
        tasks.append(
            {
                "task_description": task_description,
                "episodes": episodes,
                "successes": successes,
                "success_rate": (float(successes) / float(episodes)) if episodes else 0.0,
                "failure_category": dict(sorted(task_entry["failure_category"].items())),
            }
        )

    return {
        "task_suite_name": args["task_suite_name"],
        "num_trials_per_task": args["num_trials_per_task"],
        "max_tasks": args["max_tasks"],
        "checkpoint_path": metadata["checkpoint_path"],
        "checkpoint_hash": metadata["checkpoint_hash"],
        "config_path": metadata["config_path"],
        "config_hash": metadata["config_hash"],
        "data_version": metadata["data_version"],
        "starflow_mapping": metadata["starflow_mapping"],
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": (float(total_successes) / float(total_episodes)) if total_episodes else 0.0,
        "failure_category": dict(sorted(failure_counter.items())),
        "tasks": tasks,
        "episodes": episode_records,
    }


def write_eval_report(video_out_path: str | Path, report: dict[str, Any]) -> Path:
    output_path = Path(video_out_path) / REPORT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_eval_report(video_out_path: str | Path) -> dict[str, Any] | None:
    output_path = Path(video_out_path) / REPORT_FILENAME
    if not output_path.exists():
        return None
    return json.loads(output_path.read_text(encoding="utf-8"))
