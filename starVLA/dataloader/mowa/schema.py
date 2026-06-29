"""MoWA G0 schema skeleton.

本模块只定义数据门禁和 window 切片所需的最小结构，不读取真实数据，
也不把任何未实测 profile 写成 measured。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


DATA_GATE = "Data Gate"
TBD = "TBD"

_FORBIDDEN_WAM_INPUT_KEYS = {
    "future_action",
    "future_action_label",
    "future_actions",
    "action_chunk_target",
    "future_rgb_window",
    "future_wan_latent",
    "target_future_latent",
}


@dataclass(frozen=True)
class MoWAUnifiedEpisode:
    """MoWA episode 级统一 schema 草案。"""

    episode_id: str
    dataset_source: str
    split: str
    instruction: str
    timestamps: Sequence[float | int]
    observations: Mapping[str, Any] = field(default_factory=dict)
    actions: Mapping[str, Any] = field(default_factory=dict)
    wam_targets: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("MoWA episode_id must be non-empty.")
        if not self.dataset_source:
            raise ValueError("MoWA dataset_source must be non-empty.")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"MoWA split must be train/val/test, got {self.split!r}.")
        if not self.instruction:
            raise ValueError("MoWA instruction must be non-empty.")
        if not self.timestamps:
            raise ValueError("MoWA timestamps must be non-empty.")

        prev = self.timestamps[0]
        for idx, ts in enumerate(self.timestamps[1:], start=1):
            if ts < prev:
                raise ValueError(
                    f"MoWA timestamps must be monotonic: index {idx - 1}={prev}, index {idx}={ts}."
                )
            prev = ts

    @property
    def num_steps(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True)
class MoWAWindowConfig:
    """Episode-to-window 采样配置。

    数值只在单元测试和本地 smoke 中使用；真实数据的 Hz/window/horizon
    必须由 G0 profile 决定。
    """

    history_steps: int
    future_steps: int
    action_chunk_steps: int

    def validate(self) -> None:
        if self.history_steps <= 0:
            raise ValueError("MoWA history_steps must be positive.")
        if self.future_steps <= 0:
            raise ValueError("MoWA future_steps must be positive.")
        if self.action_chunk_steps <= 0:
            raise ValueError("MoWA action_chunk_steps must be positive.")


@dataclass(frozen=True)
class MoWAWindowSample:
    """模型 forward 级 window 样本草案。"""

    episode_id: str
    dataset_source: str
    anchor_index: int
    history_indices: tuple[int, ...]
    current_index: int
    future_indices: tuple[int, ...]
    action_target_indices: tuple[int, ...]
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    boundary_mask: Mapping[str, bool] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_boundaries(self) -> None:
        if self.current_index != self.anchor_index:
            raise ValueError("MoWA current_index must equal anchor_index.")
        if not self.history_indices:
            raise ValueError("MoWA history_indices must be non-empty.")
        if any(idx > self.anchor_index for idx in self.history_indices):
            raise ValueError("MoWA history_indices cannot include future timesteps.")
        if any(idx <= self.anchor_index for idx in self.future_indices):
            raise ValueError("MoWA future_indices must be strictly after anchor_index.")
        if any(idx < self.anchor_index for idx in self.action_target_indices):
            raise ValueError("MoWA action_target_indices cannot start before anchor_index.")

    def assert_no_future_leakage(self) -> None:
        leaked = sorted(set(self.inputs.keys()) & _FORBIDDEN_WAM_INPUT_KEYS)
        if leaked:
            raise ValueError(f"MoWA WAM inputs contain forbidden future keys: {leaked}")

    def validate(self) -> None:
        self.validate_boundaries()
        self.assert_no_future_leakage()
