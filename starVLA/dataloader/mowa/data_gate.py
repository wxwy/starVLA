"""MoWA G0 Data Verification Gate report skeleton."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from starVLA.dataloader.mowa.schema import DATA_GATE, TBD


MOWA_PRIMARY_CANDIDATE = "robocasa365"
MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH = "v1.0/target/atomic/OpenDrawer/20250816/lerobot"
MOWA_ROBOCASA365_OPEN_DRAWER_MIXTURE = "robocasa365_open_drawer_target_human"


@dataclass(frozen=True)
class MoWADataGateCandidate:
    dataset: str
    role: str
    starvla_support: str
    default_use: str
    download_status: str = DATA_GATE
    license_status: str = DATA_GATE
    schema_status: str = DATA_GATE
    temporal_profile_status: str = DATA_GATE
    p0_label_status: str = DATA_GATE
    latent_cache_status: str = DATA_GATE
    leakage_status: str = TBD
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "role": self.role,
            "starvla_support": self.starvla_support,
            "default_use": self.default_use,
            "download_status": self.download_status,
            "license_status": self.license_status,
            "schema_status": self.schema_status,
            "temporal_profile_status": self.temporal_profile_status,
            "p0_label_status": self.p0_label_status,
            "latent_cache_status": self.latent_cache_status,
            "leakage_status": self.leakage_status,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MoWADataGateReport:
    report_id: str = "mowa_g0_datagate_TBD"
    stage: str = "G0"
    experiment_budget: str = "not_counted"
    candidates: tuple[MoWADataGateCandidate, ...] = field(default_factory=tuple)
    local_checks: Mapping[str, Any] = field(default_factory=dict)
    go_no_go: str = TBD
    unresolved_items: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "stage": self.stage,
            "experiment_budget": self.experiment_budget,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "local_checks": dict(self.local_checks),
            "go_no_go": self.go_no_go,
            "unresolved_items": list(self.unresolved_items),
        }


def build_mowa_g0_report_skeleton() -> MoWADataGateReport:
    """生成不含实测数值的 G0 报告骨架。"""

    return MoWADataGateReport(
        candidates=(
            MoWADataGateCandidate(
                dataset="robocasa365",
                role="PrimaryCandidate",
                starvla_support="train+eval+registry",
                default_use="G0/P0 first loop",
                notes="先用 OpenDrawer target/human 做最小闭环；G0 通过后再扩展 target_human_all。",
            ),
            MoWADataGateCandidate(
                dataset="libero",
                role="AuxiliarySanity",
                starvla_support="train+eval+registry",
                default_use="action schema and sampler sanity",
                notes="固定桌面操作辅助集，不能作为移动操作主证据。",
            ),
            MoWADataGateCandidate(
                dataset="robotwin",
                role="AuxiliarySanity",
                starvla_support="train+eval+registry",
                default_use="action bridge and manipulation sanity",
                notes="操作泛化和双臂 sanity，不作为 MoWA 第一主闭环。",
            ),
        ),
        unresolved_items=(
            "obs fps / action Hz must be profiled by G0 before use.",
            "history window / future horizon / action chunk must remain Data Gate before profiling.",
            "future action leakage tests must pass before P0/P1 training.",
        ),
    )


def build_mowa_robocasa365_local_smoke_report(
    data_root: Path | str,
    mixture_name: str = MOWA_ROBOCASA365_OPEN_DRAWER_MIXTURE,
) -> MoWADataGateReport:
    """检查 RoboCasa365 最小闭环数据路径是否已在本地准备。

    该函数只做路径存在性检查，不读取真实数据，不产生 fps/Hz/window
    等 profile 结论。
    """

    root = Path(data_root)
    expected_relative_path = MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
    expected_path = root / expected_relative_path
    path_exists = expected_path.exists()

    meta_checks = _inspect_robocasa365_lerobot_meta(expected_path) if path_exists else {}
    schema_available = bool(meta_checks.get("schema_smoke_available", False))

    report = build_mowa_g0_report_skeleton()
    candidates = []
    for candidate in report.candidates:
        if candidate.dataset == MOWA_PRIMARY_CANDIDATE:
            candidates.append(
                replace(
                    candidate,
                    download_status="available" if path_exists else "missing",
                    schema_status="schema_smoke_available" if schema_available else DATA_GATE,
                    temporal_profile_status=DATA_GATE,
                    leakage_status=TBD,
                    notes=(
                        "本地已发现 OpenDrawer target/human 最小闭环路径；"
                        "已完成 meta 只读 schema smoke，仍需执行 profile/leakage。"
                        if schema_available
                        else "本地已发现 OpenDrawer target/human 最小闭环路径；仍需执行 schema/profile/leakage。"
                        if path_exists
                        else "本地未发现 OpenDrawer target/human 最小闭环路径；需先准备数据后再做真实 G0。"
                    ),
                )
            )
        else:
            candidates.append(candidate)

    return replace(
        report,
        candidates=tuple(candidates),
        local_checks={
            "dataset": MOWA_PRIMARY_CANDIDATE,
            "mixture_name": mixture_name,
            "data_root": str(root),
            "expected_relative_path": expected_relative_path,
            "expected_path": str(expected_path),
            "path_exists": path_exists,
            **meta_checks,
            "profile_status": DATA_GATE,
        },
        go_no_go=(
            "No-Go: schema smoke available; profile/leakage pending"
            if schema_available
            else "TBD"
            if path_exists
            else "No-Go: missing local minimal dataset"
        ),
    )


def _inspect_robocasa365_lerobot_meta(dataset_path: Path) -> dict[str, Any]:
    """只读检查 Lerobot meta，不读取 parquet/video 内容。"""

    meta_dir = dataset_path / "meta"
    extras_dir = dataset_path / "extras"
    data_dir = dataset_path / "data"
    video_dir = dataset_path / "videos"

    required_meta_files = (
        "info.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "tasks.jsonl",
        "modality.json",
    )
    existing_meta_files = tuple(
        name for name in required_meta_files if (meta_dir / name).is_file()
    )
    missing_meta_files = tuple(
        name for name in required_meta_files if name not in existing_meta_files
    )

    info = _read_json(meta_dir / "info.json")
    modality = _read_json(meta_dir / "modality.json")
    episodes = _read_jsonl(meta_dir / "episodes.jsonl", limit=5)

    features = info.get("features", {}) if isinstance(info, dict) else {}
    modality_video = modality.get("video", {}) if isinstance(modality, dict) else {}
    modality_state = modality.get("state", {}) if isinstance(modality, dict) else {}
    modality_action = modality.get("action", {}) if isinstance(modality, dict) else {}

    required_feature_keys = (
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
    )
    existing_feature_keys = tuple(key for key in required_feature_keys if key in features)
    missing_feature_keys = tuple(
        key for key in required_feature_keys if key not in existing_feature_keys
    )
    video_feature_keys = tuple(
        key for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "video"
    )

    return {
        "schema_smoke_available": not missing_meta_files and not missing_feature_keys,
        "meta_dir": str(meta_dir),
        "existing_meta_files": existing_meta_files,
        "missing_meta_files": missing_meta_files,
        "declared_total_episodes": info.get("total_episodes", TBD),
        "declared_total_frames": info.get("total_frames", TBD),
        "declared_fps": info.get("fps", DATA_GATE),
        "declared_splits": info.get("splits", TBD),
        "declared_robot_type": info.get("robot_type", TBD),
        "declared_data_path_template": info.get("data_path", TBD),
        "declared_video_path_template": info.get("video_path", TBD),
        "feature_keys_present": existing_feature_keys,
        "feature_keys_missing": missing_feature_keys,
        "video_feature_keys": video_feature_keys,
        "modality_video_keys": tuple(modality_video.keys()),
        "modality_state_keys": tuple(modality_state.keys()),
        "modality_action_keys": tuple(modality_action.keys()),
        "sample_episodes": episodes,
        "data_dir_exists": data_dir.is_dir(),
        "video_dir_exists": video_dir.is_dir(),
        "extras_dir_exists": extras_dir.is_dir(),
        "parquet_file_count": _count_files(data_dir, "*.parquet"),
        "video_file_count": _count_files(video_dir, "*.mp4"),
        "episode_extra_dir_count": _count_dirs(extras_dir, "episode_*"),
        "action_hz_status": DATA_GATE,
        "obs_fps_status": DATA_GATE,
        "history_window_status": DATA_GATE,
        "future_window_status": DATA_GATE,
        "leakage_check_status": TBD,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path, limit: int) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
    return tuple(rows)


def _count_files(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob(pattern) if path.is_file())


def _count_dirs(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for path in root.glob(pattern) if path.is_dir())
