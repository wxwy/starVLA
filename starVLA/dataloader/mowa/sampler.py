"""MoWA episode-to-window sampler skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from starVLA.dataloader.mowa.schema import (
    DATA_GATE,
    MoWAUnifiedEpisode,
    MoWAWindowConfig,
    MoWAWindowSample,
)


@dataclass(frozen=True)
class MoWAEpisodeToWindowSampler:
    """从完整 episode 中切出 MoWA WindowSample。

    完整 episode 只是采样容器；future action 只进入 action target，
    不进入 WAM inputs。
    """

    config: MoWAWindowConfig

    def __post_init__(self) -> None:
        self.config.validate()

    def sample(self, episode: MoWAUnifiedEpisode, anchor_index: int) -> MoWAWindowSample:
        episode.validate()
        if anchor_index < 0 or anchor_index >= episode.num_steps:
            raise IndexError(
                f"MoWA anchor_index out of range: {anchor_index}, episode length={episode.num_steps}."
            )

        hist_start = max(0, anchor_index - self.config.history_steps + 1)
        history_indices = tuple(range(hist_start, anchor_index + 1))

        future_end = min(episode.num_steps, anchor_index + self.config.future_steps + 1)
        future_indices = tuple(range(anchor_index + 1, future_end))

        action_end = min(episode.num_steps, anchor_index + self.config.action_chunk_steps)
        action_target_indices = tuple(range(anchor_index, action_end))

        inputs = {
            "current_index": anchor_index,
            "history_indices": history_indices,
            "instruction": episode.instruction,
            "observations": episode.observations,
            "history_actions": {
                "indices": history_indices,
                "source": "executed_history_action",
            },
        }
        targets = {
            "future_indices": future_indices,
            "action_chunk_target": {
                "indices": action_target_indices,
                "source": "future action target only",
            },
            "wam_targets": episode.wam_targets,
        }
        boundary_mask = {
            "has_full_history": len(history_indices) == self.config.history_steps,
            "has_full_future": len(future_indices) == self.config.future_steps,
            "has_full_action_chunk": len(action_target_indices) == self.config.action_chunk_steps,
        }
        metadata = {
            "obs_fps": episode.metadata.get("obs_fps", DATA_GATE),
            "action_hz": episode.metadata.get("action_hz", DATA_GATE),
            "history_steps": self.config.history_steps,
            "future_steps": self.config.future_steps,
            "action_chunk_steps": self.config.action_chunk_steps,
        }

        sample = MoWAWindowSample(
            episode_id=episode.episode_id,
            dataset_source=episode.dataset_source,
            anchor_index=anchor_index,
            history_indices=history_indices,
            current_index=anchor_index,
            future_indices=future_indices,
            action_target_indices=action_target_indices,
            inputs=inputs,
            targets=targets,
            boundary_mask=boundary_mask,
            metadata=metadata,
        )
        sample.validate()
        return sample


def select_mowa_smoke_anchor_index(row_count: int, window_config: MoWAWindowConfig) -> int:
    """Select the first anchor with full history when the episode is long enough."""

    if row_count <= 0:
        return 0
    first_full_history_anchor = max(window_config.history_steps - 1, 0)
    last_full_future_anchor = max(row_count - window_config.future_steps - 1, 0)
    return min(first_full_history_anchor, last_full_future_anchor, row_count - 1)


def select_mowa_leakage_anchor_indices(
    row_count: int,
    window_config: MoWAWindowConfig,
) -> tuple[int, ...]:
    """Select start/mid/end anchors for metadata-level leakage checks."""

    if row_count <= 0:
        return ()
    start = select_mowa_smoke_anchor_index(row_count, window_config)
    mid = max(0, min(row_count - 1, row_count // 2))
    end = max(0, min(row_count - 1, row_count - window_config.future_steps - 1))
    return tuple(sorted({start, mid, end}))
