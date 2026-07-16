"""MoWA window-level latent sample assembler.

Assembles a ``WindowLatentSample`` from a window manifest entry and an episode
latent store.  Performs index validation and leakage checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from starVLA.dataloader.mowa.episode_latent_store import MoWAEpisodeLatentStore
from starVLA.dataloader.mowa.instruction_text_latent_cache import (
    MoWAInstructionTextLatentCache,
)
from starVLA.dataloader.mowa.schema import _FORBIDDEN_WAM_INPUT_KEYS
from starVLA.dataloader.mowa.window_manifest import (
    MoWAWindowManifestEntry,
    MoWAWindowManifestTable,
    load_mowa_window_manifest_table,
    slice_mowa_window_manifest_entry,
)


@dataclass(frozen=True)
class WindowLatentSample:
    """Unified sample returned by the episode-level latent dataloader."""

    episode_id: str
    sample_id: str
    anchor_index: int

    current_latent: torch.Tensor
    history_latents: torch.Tensor
    future_latents: torch.Tensor
    multi_view_video_keys: tuple[str, ...]
    multi_view_history_latents: torch.Tensor
    multi_view_current_latents: torch.Tensor
    multi_view_future_latents: torch.Tensor

    robot_state_history: torch.Tensor
    current_robot_state: torch.Tensor
    future_robot_states: torch.Tensor
    history_actions: torch.Tensor
    action_chunk: torch.Tensor

    history_valid_mask: torch.Tensor
    future_valid_mask: torch.Tensor
    action_valid_mask: torch.Tensor
    history_action_valid_mask: torch.Tensor
    history_state_valid_mask: torch.Tensor
    future_state_valid_mask: torch.Tensor
    future_done_target: torch.Tensor

    language: str | dict[str, Any]
    labels: dict[str, Any]
    metadata: dict[str, Any]


class MoWAWindowLatentSampleDataset:
    """Dataset that reads window manifest and gathers latents from episode stores."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        label_sidecar_root: Path | str | None = None,
        instruction_text_latent: Path | str | MoWAInstructionTextLatentCache | None = None,
        history_steps: int | None = None,
        future_steps: int | None = None,
        action_chunk_steps: int | None = None,
        episode_latent_path_prefix: Path | str | None = None,
        video_keys: tuple[str, ...] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if action_chunk_steps is None and future_steps is not None:
            action_chunk_steps = int(future_steps) * 4
        self.label_sidecar_root = (
            Path(label_sidecar_root) if label_sidecar_root is not None else None
        )
        self.video_keys = tuple(video_keys or ())
        self._instruction_table: MoWAInstructionTextLatentCache | None = None
        if instruction_text_latent is not None:
            if isinstance(instruction_text_latent, MoWAInstructionTextLatentCache):
                self._instruction_table = instruction_text_latent
            else:
                self._instruction_table = MoWAInstructionTextLatentCache(
                    instruction_text_latent
        )
        self._history_steps = history_steps
        self._future_steps = future_steps
        self._action_chunk_steps = action_chunk_steps
        self._entries: MoWAWindowManifestTable = load_mowa_window_manifest_table(
            self.manifest_path,
            episode_latent_path_prefix=episode_latent_path_prefix,
        )
        self._store_cache: dict[str, MoWAEpisodeLatentStore] = {}
        self._raw_episode_cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> WindowLatentSample:
        entry = self.get_entry(index)
        return self._assemble_sample(entry)

    def get_entry(self, index: int) -> MoWAWindowManifestEntry:
        return slice_mowa_window_manifest_entry(
            self._entries[index],
            history_steps=self._history_steps,
            future_steps=self._future_steps,
            action_chunk_steps=self._action_chunk_steps,
        )

    def sample_keys(self) -> tuple[tuple[int, int, str], ...]:
        return self._entries.sample_keys()

    def _get_store(self, path: str) -> MoWAEpisodeLatentStore:
        store = self._store_cache.get(path)
        if store is None:
            store = MoWAEpisodeLatentStore(Path(path))
            self._store_cache[path] = store
        return store

    def _load_language(self, entry: MoWAWindowManifestEntry) -> str | dict[str, Any]:
        """Return raw instruction string or cached text latent dict."""
        store = self._get_store(entry.episode_latent_path)
        instruction = store.get_language()

        if self._instruction_table is not None:
            cached = self._instruction_table.lookup(instruction)
            if cached is not None:
                return {
                    "instruction": instruction,
                    "text_embeds": cached["text_embeds"],
                    "attention_mask": cached["attention_mask"],
                    "pooled_text_hidden": cached["pooled_text_hidden"],
                }

        return instruction

    def _load_labels(self, entry: MoWAWindowManifestEntry) -> dict[str, Any]:
        if entry.label_sidecar_path is None:
            return {}
        sidecar_path = Path(entry.label_sidecar_path)
        if self.label_sidecar_root is not None and not sidecar_path.is_absolute():
            sidecar_path = self.label_sidecar_root / sidecar_path
        if not sidecar_path.is_file():
            sidecar_path = self._legacy_jsonl_sidecar_path(sidecar_path)
            if not sidecar_path.is_file():
                return {}
        if sidecar_path.suffix == ".parquet":
            return self._load_label_parquet(sidecar_path, entry.label_index)
        try:
            import json

            with sidecar_path.open("r", encoding="utf-8") as file:
                lines = [line.strip() for line in file if line.strip()]
            if entry.label_index < len(lines):
                return json.loads(lines[entry.label_index])
        except Exception:  # noqa: BLE001
            pass
        return {}

    def _legacy_jsonl_sidecar_path(self, sidecar_path: Path) -> Path:
        if sidecar_path.suffix != ".parquet":
            return sidecar_path
        stem = sidecar_path.stem
        if stem.startswith("episode_"):
            return sidecar_path.with_name(f"ep_{stem.split('_')[-1]}.jsonl")
        return sidecar_path.with_suffix(".jsonl")

    def _load_label_parquet(self, sidecar_path: Path, label_index: int) -> dict[str, Any]:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(sidecar_path)
            if label_index >= table.num_rows:
                return {}
            row = table.slice(label_index, 1).to_pydict()
            return {
                key: values[0]
                for key, values in row.items()
                if values and values[0] is not None
            }
        except Exception:  # noqa: BLE001
            return {}

    def _get_raw_episode_data(self, entry: MoWAWindowManifestEntry) -> Any | None:
        raw_episode_path = entry.source_episode_path
        if not raw_episode_path:
            return None
        cached = self._raw_episode_cache.get(raw_episode_path)
        if cached is not None:
            return cached
        path = Path(raw_episode_path)
        if not path.is_file():
            return None
        try:
            raw_episode = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return None
        self._raw_episode_cache[raw_episode_path] = raw_episode
        return raw_episode

    def _load_raw_sequence(
        self,
        raw_episode: Any,
        column_name: str,
        indices: tuple[int, ...],
    ) -> torch.Tensor | None:
        if raw_episode is None or column_name not in raw_episode.columns:
            return None
        try:
            values = [raw_episode[column_name].iloc[index] for index in indices]
            return torch.from_numpy(np.stack([np.asarray(value) for value in values]))
        except Exception:  # noqa: BLE001
            return None

    def _load_raw_language(self, raw_episode: Any, anchor_index: int) -> str | None:
        if raw_episode is None:
            return None
        for column_name in ("instruction", "lang", "language", "task"):
            if column_name not in raw_episode.columns:
                continue
            try:
                value = raw_episode[column_name].iloc[min(anchor_index, len(raw_episode) - 1)]
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, str) and value:
                return value
        return None

    def _assemble_sample(self, entry: MoWAWindowManifestEntry) -> WindowLatentSample:
        validate_window_indices(entry)

        store = self._get_store(entry.episode_latent_path)
        raw_episode = self._get_raw_episode_data(entry)

        is_wan_regular_grid = int(store.attrs.get("temporal_compression_factor", 1)) == 4
        get_latents = store.get_latents_by_latent_indices if is_wan_regular_grid else store.get_latents
        current_latent = get_latents(entry.anchor_video_key, (entry.current_index,))
        history_latents = get_latents(entry.anchor_video_key, entry.history_indices)
        future_latents = get_latents(entry.anchor_video_key, entry.future_indices)
        multi_view_video_keys = self.video_keys or (entry.anchor_video_key,)
        available_video_keys = set(store.list_video_keys())
        missing_video_keys = [key for key in multi_view_video_keys if key not in available_video_keys]
        if missing_video_keys:
            raise KeyError(
                f"Episode {entry.episode_id} is missing requested latent views: {missing_video_keys}."
            )
        anchor_source_indices = store.get_latent_source_frame_indices(entry.anchor_video_key)
        for video_key in multi_view_video_keys:
            source_indices = store.get_latent_source_frame_indices(video_key)
            if (
                anchor_source_indices is not None
                and source_indices is not None
                and not np.array_equal(anchor_source_indices, source_indices)
            ):
                raise ValueError(
                    f"Multi-view latent timestamps are not aligned for episode={entry.episode_id}: "
                    f"anchor_view={entry.anchor_video_key}, mismatched_view={video_key}."
                )
        multi_view_history_latents = torch.stack(
            [get_latents(video_key, entry.history_indices) for video_key in multi_view_video_keys], dim=0
        )
        multi_view_current_latents = torch.stack(
            [get_latents(video_key, (entry.current_index,)).squeeze(0) for video_key in multi_view_video_keys],
            dim=0,
        )
        multi_view_future_latents = torch.stack(
            [get_latents(video_key, entry.future_indices) for video_key in multi_view_video_keys], dim=0
        )
        last_latent_index = store.num_latent_frames(entry.anchor_video_key) - 1
        future_done_target = _future_done_target(entry.future_indices, last_latent_index)

        # Squeeze the single-frame current latent if it is pooled vector form.
        if current_latent.shape[0] == 1:
            current_latent = current_latent.squeeze(0)

        robot_state_history = self._load_raw_sequence(raw_episode, "observation.state", entry.history_state_indices)
        current_robot_state = self._load_raw_sequence(
            raw_episode, "observation.state", (entry.current_state_index,)
        )
        future_robot_states = self._load_raw_sequence(
            raw_episode, "observation.state", entry.future_state_indices
        )
        history_actions = self._load_raw_sequence(raw_episode, "action", entry.history_action_indices)
        action_chunk = self._load_raw_sequence(raw_episode, "action", entry.action_chunk_indices)

        if (
            robot_state_history is None
            or current_robot_state is None
            or future_robot_states is None
            or history_actions is None
            or action_chunk is None
        ):
            robot_state = store.get_robot_state()
            robot_action = store.get_robot_action()
            if robot_state_history is None:
                robot_state_history = robot_state[list(entry.history_state_indices), ...]
            if current_robot_state is None:
                current_robot_state = robot_state[entry.current_state_index, ...]
            if future_robot_states is None:
                future_robot_states = robot_state[list(entry.future_state_indices), ...]
            if history_actions is None:
                history_actions = robot_action[list(entry.history_action_indices), ...]
            if action_chunk is None:
                action_chunk = robot_action[list(entry.action_chunk_indices), ...]

        current_robot_state = current_robot_state.squeeze(0)

        history_actions = _apply_action_padding(
            history_actions,
            entry.history_action_valid_mask,
            entry.action_representation,
        )
        action_chunk = _apply_action_padding(
            action_chunk,
            entry.action_valid_mask,
            entry.action_representation,
        )

        raw_language = self._load_raw_language(raw_episode, entry.anchor_index)
        language = raw_language if raw_language is not None else self._load_language(entry)
        labels = self._load_labels(entry)

        metadata = {
            "episode_id": entry.episode_id,
            "sample_id": entry.sample_id,
            "anchor_index": entry.anchor_index,
            "anchor_video_key": entry.anchor_video_key,
            "history_indices": entry.history_indices,
            "current_index": entry.current_index,
            "future_indices": entry.future_indices,
            "history_state_indices": entry.history_state_indices,
            "current_state_index": entry.current_state_index,
            "future_state_indices": entry.future_state_indices,
            "history_action_indices": entry.history_action_indices,
            "action_chunk_indices": entry.action_chunk_indices,
            "history_valid_mask": entry.history_valid_mask,
            "future_valid_mask": entry.future_valid_mask,
            "action_valid_mask": entry.action_valid_mask,
            "history_action_valid_mask": entry.history_action_valid_mask,
            "history_state_valid_mask": entry.history_state_valid_mask,
            "future_state_valid_mask": entry.future_state_valid_mask,
            "action_representation": entry.action_representation,
            "wam_hz": entry.wam_hz,
            "history_stride": entry.history_stride,
            "source_dataset_path": entry.source_dataset_path,
            "source_episode_path": entry.source_episode_path,
        }

        sample = WindowLatentSample(
            episode_id=entry.episode_id,
            sample_id=entry.sample_id,
            anchor_index=entry.anchor_index,
            current_latent=current_latent,
            history_latents=history_latents,
            future_latents=future_latents,
            multi_view_video_keys=multi_view_video_keys,
            multi_view_history_latents=multi_view_history_latents,
            multi_view_current_latents=multi_view_current_latents,
            multi_view_future_latents=multi_view_future_latents,
            robot_state_history=robot_state_history,
            current_robot_state=current_robot_state,
            future_robot_states=future_robot_states,
            history_actions=history_actions,
            action_chunk=action_chunk,
            history_valid_mask=torch.tensor(entry.history_valid_mask, dtype=torch.bool),
            future_valid_mask=torch.tensor(entry.future_valid_mask, dtype=torch.bool),
            action_valid_mask=torch.tensor(entry.action_valid_mask, dtype=torch.bool),
            history_action_valid_mask=torch.tensor(entry.history_action_valid_mask, dtype=torch.bool),
            history_state_valid_mask=torch.tensor(entry.history_state_valid_mask, dtype=torch.bool),
            future_state_valid_mask=torch.tensor(entry.future_state_valid_mask, dtype=torch.bool),
            future_done_target=future_done_target,
            language=language,
            labels=labels,
            metadata=metadata,
        )
        assert_no_future_leakage(sample)
        return sample



def validate_window_indices(entry: MoWAWindowManifestEntry) -> None:
    """Validate temporal boundaries of a window manifest entry."""

    # Wan 的 anchor_index 是原始 state/action 帧索引，而 current_index 是
    # regular latent 索引；二者不要求数值相等。
    if entry.current_state_index != entry.anchor_index:
        raise ValueError(
            f"current_state_index must equal anchor_index: "
            f"{entry.current_state_index} != {entry.anchor_index}"
        )
    if entry.history_indices and max(entry.history_indices) > entry.current_index:
        raise ValueError(
            f"history_indices cannot include future timesteps: max={max(entry.history_indices)}, "
            f"current={entry.current_index}"
        )
    valid_history_indices = (
        tuple(index for index, valid in zip(entry.history_indices, entry.history_valid_mask) if valid)
        if entry.history_valid_mask
        else ()
    )
    if valid_history_indices and max(valid_history_indices) >= entry.current_index:
        raise ValueError(
            f"history_indices must be strictly before current_index: max={max(entry.history_indices)}, "
            f"current={entry.current_index}"
        )
    if not entry.future_indices:
        raise ValueError("future_indices must be non-empty.")
    valid_future_indices = (
        tuple(index for index, valid in zip(entry.future_indices, entry.future_valid_mask) if valid)
        if entry.future_valid_mask
        else entry.future_indices
    )
    if valid_future_indices and min(valid_future_indices) <= entry.current_index:
        raise ValueError(
            f"future_indices must be strictly after current_index: min={min(entry.future_indices)}, "
            f"current={entry.current_index}"
        )
    valid_future_state_indices = (
        tuple(index for index, valid in zip(entry.future_state_indices, entry.future_state_valid_mask) if valid)
        if entry.future_state_valid_mask
        else entry.future_state_indices
    )
    if valid_future_state_indices and min(valid_future_state_indices) <= entry.current_state_index:
        raise ValueError(
            f"future_state_indices must be strictly after current_state_index: "
            f"min={min(valid_future_state_indices)}, current={entry.current_state_index}"
        )
    if not _is_monotonic(entry.history_indices, allow_repeated=bool(entry.history_valid_mask)):
        raise ValueError(f"history_indices must be monotonic: {entry.history_indices}")
    if not _is_monotonic(entry.future_indices, allow_repeated=bool(entry.future_valid_mask)):
        raise ValueError(f"future_indices must be monotonic: {entry.future_indices}")
    valid_action_indices = (
        tuple(index for index, valid in zip(entry.action_chunk_indices, entry.action_valid_mask) if valid)
        if entry.action_valid_mask
        else ()
    )
    if valid_action_indices and min(valid_action_indices) <= entry.anchor_index:
        raise ValueError(
            f"action_chunk_indices must start after anchor_index: "
            f"min={min(entry.action_chunk_indices)}, anchor={entry.anchor_index}"
        )
    for name, indices, mask in (
        ("history", entry.history_indices, entry.history_valid_mask),
        ("history_state", entry.history_state_indices, entry.history_state_valid_mask),
        ("future_state", entry.future_state_indices, entry.future_state_valid_mask),
        ("future", entry.future_indices, entry.future_valid_mask),
        ("action", entry.action_chunk_indices, entry.action_valid_mask),
        ("history_action", entry.history_action_indices, entry.history_action_valid_mask),
    ):
        if mask and len(mask) != len(indices):
            raise ValueError(f"{name}_valid_mask length must match {name} indices.")


def assert_no_future_leakage(sample: WindowLatentSample) -> None:
    """Ensure no forbidden future keys appear in the assembled sample inputs."""

    inputs = {
        "current_latent",
        "history_latents",
        "robot_state_history",
        "history_actions",
        "language",
    }
    leaked = sorted(set(inputs) & _FORBIDDEN_WAM_INPUT_KEYS)
    if leaked:
        raise ValueError(f"WindowLatentSample inputs contain forbidden future keys: {leaked}")

    # Explicit structural checks: future latents / action chunk / labels are not inputs.
    for key in ("future_latents", "action_chunk", "labels"):
        if key in inputs:
            raise ValueError(f"Forbidden key {key!r} appears in sample inputs.")


def _is_monotonic(indices: tuple[int, ...], *, allow_repeated: bool = False) -> bool:
    if len(indices) <= 1:
        return True
    if allow_repeated:
        return all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))
    return all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))


def _apply_action_padding(
    actions: torch.Tensor,
    valid_mask: tuple[bool, ...],
    action_representation: str,
) -> torch.Tensor:
    """Zero invalid delta actions; absolute actions already use boundary repeats."""
    if action_representation == "absolute" or not valid_mask:
        return actions
    if action_representation != "delta":
        raise ValueError(f"Unsupported action_representation: {action_representation!r}")
    result = actions.clone()
    result[~torch.tensor(valid_mask, dtype=torch.bool)] = 0
    return result


def _future_done_target(
    future_indices: tuple[int, ...],
    last_latent_index: int,
) -> torch.Tensor:
    """Mark the first future position reaching episode terminal and all later positions."""
    reached_terminal = False
    targets: list[float] = []
    for index in future_indices:
        reached_terminal = reached_terminal or index >= last_latent_index
        targets.append(float(reached_terminal))
    return torch.tensor(targets, dtype=torch.float32)
