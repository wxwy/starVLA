"""MoWA G0 Data Verification Gate report skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from starVLA.dataloader.mowa.schema import DATA_GATE, TBD


MOWA_PRIMARY_CANDIDATE = "robocasa365"


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
    go_no_go: str = TBD
    unresolved_items: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "stage": self.stage,
            "experiment_budget": self.experiment_budget,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
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
