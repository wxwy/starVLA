"""MoWA G0 Data Verification Gate report skeleton."""

from __future__ import annotations

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

    report = build_mowa_g0_report_skeleton()
    candidates = []
    for candidate in report.candidates:
        if candidate.dataset == MOWA_PRIMARY_CANDIDATE:
            candidates.append(
                replace(
                    candidate,
                    download_status="available" if path_exists else "missing",
                    schema_status=DATA_GATE,
                    temporal_profile_status=DATA_GATE,
                    leakage_status=TBD,
                    notes=(
                        "本地已发现 OpenDrawer target/human 最小闭环路径；仍需执行 schema/profile/leakage。"
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
            "profile_status": DATA_GATE,
        },
        go_no_go="TBD" if path_exists else "No-Go: missing local minimal dataset",
    )
