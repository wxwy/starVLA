"""MoWA episode-to-window sampler skeleton."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch

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


def build_mowa_shuffled_episode_pairs(
    episode_indices: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Build deterministic non-self shuffled pairs for robot-history sanity checks."""

    if len(episode_indices) <= 1:
        return ()
    return tuple(
        (episode_index, episode_indices[(idx + 1) % len(episode_indices)])
        for idx, episode_index in enumerate(episode_indices)
    )


@dataclass(frozen=True)
class MoWALatentSamplingConfig:
    """Epoch-level sampling policy over the complete window manifest."""

    samples_per_episode_per_epoch: int = 1
    regular_anchor_ratio: float = 0.80
    terminal_anchor_ratio: float = 0.15
    early_anchor_ratio: float = 0.05
    replacement: bool = False
    seed: int = 42
    history_steps: int | None = None
    future_steps: int | None = None
    action_chunk_steps: int | None = None

    def validate(self) -> None:
        if self.samples_per_episode_per_epoch <= 0:
            raise ValueError("samples_per_episode_per_epoch must be positive.")
        if self.history_steps is not None and self.history_steps < 0:
            raise ValueError("history_steps must be non-negative when set.")
        if self.future_steps is not None and self.future_steps <= 0:
            raise ValueError("future_steps must be positive when set.")
        if self.action_chunk_steps is not None and self.action_chunk_steps <= 0:
            raise ValueError("action_chunk_steps must be positive when set.")
        ratios = (self.regular_anchor_ratio, self.terminal_anchor_ratio, self.early_anchor_ratio)
        if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("regular/terminal/early anchor ratios must be non-negative and sum to 1.")


@dataclass(frozen=True)
class _SamplingManifestEntry:
    task_name: str
    episode_id: str
    anchor_index: int
    history_indices: Sequence[int]
    history_valid_mask: Sequence[bool]
    history_action_valid_mask: Sequence[bool]
    future_valid_mask: Sequence[bool]
    action_valid_mask: Sequence[bool]


class MoWAManifestAnchorSampler(torch.utils.data.Sampler[int]):
    """Deterministic task→episode→anchor sampler over manifest-backed rows."""

    def __init__(
        self,
        dataset_steps: Sequence[tuple[int, int]],
        entries: Sequence[Any],
        config: MoWALatentSamplingConfig,
    ) -> None:
        config.validate()
        self.config = config
        self.epoch = 0
        step_to_index = {tuple(map(int, step)): index for index, step in enumerate(dataset_steps)}
        self._groups: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        sampling_rows = getattr(entries, "iter_sampling_rows", None)
        if sampling_rows is not None:
            entry_rows = (
                _SamplingManifestEntry(*row)
                for row in sampling_rows()
            )
        else:
            entry_rows = entries
        for entry in entry_rows:
            try:
                episode_index = int(str(entry.episode_id).split("_")[-1])
            except ValueError:
                continue
            dataset_index = step_to_index.get((episode_index, int(entry.anchor_index)))
            if dataset_index is None:
                continue
            category = _manifest_anchor_category(
                entry,
                history_steps=self.config.history_steps,
                future_steps=self.config.future_steps,
                action_chunk_steps=self.config.action_chunk_steps,
            )
            self._groups[str(entry.task_name)][str(entry.episode_id)][category].append(dataset_index)
        self._episode_count = sum(len(episodes) for episodes in self._groups.values())
        if self._episode_count == 0:
            raise ValueError("No manifest entries map to dataset steps for MoWA sampling.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._episode_count * self.config.samples_per_episode_per_epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.config.seed + self.epoch)
        quotas = _category_quotas(len(self), self.config)
        selected: list[int] = []
        used: set[int] = set()
        episode_draws: dict[tuple[str, str], int] = defaultdict(int)
        for category, quota in quotas.items():
            for _ in range(quota):
                candidates = _category_groups(self._groups, category)
                candidates = _with_episode_capacity(candidates, episode_draws, self.config.samples_per_episode_per_epoch)
                if not self.config.replacement:
                    candidates = _without_used(candidates, used)
                if not candidates:
                    candidates = _category_groups(self._groups, None)
                    candidates = _with_episode_capacity(candidates, episode_draws, self.config.samples_per_episode_per_epoch)
                    if not self.config.replacement:
                        candidates = _without_used(candidates, used)
                if not candidates:
                    break
                task_names = sorted(candidates)
                task = task_names[torch.randint(len(task_names), (1,), generator=generator).item()]
                episode_names = sorted(candidates[task])
                episode = episode_names[torch.randint(len(episode_names), (1,), generator=generator).item()]
                anchors = candidates[task][episode]
                index = anchors[torch.randint(len(anchors), (1,), generator=generator).item()]
                selected.append(index)
                used.add(index)
                episode_draws[(task, episode)] += 1
        permutation = torch.randperm(len(selected), generator=generator).tolist()
        return iter([selected[index] for index in permutation])


def _manifest_anchor_category(
    entry: Any,
    *,
    history_steps: int | None = None,
    future_steps: int | None = None,
    action_chunk_steps: int | None = None,
) -> str:
    history_valid_mask = tuple(getattr(entry, "history_valid_mask", ()))
    history_action_valid_mask = tuple(getattr(entry, "history_action_valid_mask", ()))
    future_valid_mask = tuple(getattr(entry, "future_valid_mask", ()))
    action_valid_mask = tuple(getattr(entry, "action_valid_mask", ()))
    if history_steps is not None:
        history_valid_mask = history_valid_mask[-history_steps:] if history_steps else ()
        action_stride = len(history_action_valid_mask) // max(1, len(getattr(entry, "history_indices", ())))
        history_action_valid_mask = (
            history_action_valid_mask[-history_steps * action_stride:] if history_steps else ()
        )
    if future_steps is not None:
        future_valid_mask = future_valid_mask[:future_steps]
    if action_chunk_steps is not None:
        action_valid_mask = action_valid_mask[:action_chunk_steps]
    elif future_steps is not None:
        action_valid_mask = action_valid_mask[: future_steps * 4]

    if any(not bool(value) for value in future_valid_mask) or any(
        not bool(value) for value in action_valid_mask
    ):
        return "terminal"
    if any(not bool(value) for value in history_valid_mask) or any(
        not bool(value) for value in history_action_valid_mask
    ):
        return "early"
    return "regular"


def _category_groups(groups, category: str | None):
    result = defaultdict(dict)
    for task, episodes in groups.items():
        for episode, categories in episodes.items():
            anchors = sum(categories.values(), []) if category is None else categories.get(category, [])
            if anchors:
                result[task][episode] = anchors
    return result


def _category_quotas(total: int, config: MoWALatentSamplingConfig) -> dict[str, int]:
    raw = {
        "regular": total * config.regular_anchor_ratio,
        "terminal": total * config.terminal_anchor_ratio,
        "early": total * config.early_anchor_ratio,
    }
    quotas = {name: int(value) for name, value in raw.items()}
    for name, _ in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True)[: total - sum(quotas.values())]:
        quotas[name] += 1
    return quotas


def _with_episode_capacity(groups, episode_draws, capacity: int):
    result = defaultdict(dict)
    for task, episodes in groups.items():
        for episode, anchors in episodes.items():
            if episode_draws[(task, episode)] < capacity:
                result[task][episode] = anchors
    return result


def _without_used(groups, used: set[int]):
    result = defaultdict(dict)
    for task, episodes in groups.items():
        for episode, anchors in episodes.items():
            available = [index for index in anchors if index not in used]
            if available:
                result[task][episode] = available
    return result
