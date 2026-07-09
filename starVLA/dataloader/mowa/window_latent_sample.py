"""MoWA window-level latent sample assembler.

Assembles a ``WindowLatentSample`` from a window manifest entry and an episode
latent store.  Performs index validation and leakage checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from starVLA.dataloader.mowa.episode_latent_store import MoWAEpisodeLatentStore
from starVLA.dataloader.mowa.instruction_text_latent_cache import (
    MoWAInstructionTextLatentCache,
)
from starVLA.dataloader.mowa.schema import _FORBIDDEN_WAM_INPUT_KEYS
from starVLA.dataloader.mowa.text_latent_store import MoWATextLatentStore
from starVLA.dataloader.mowa.window_manifest import (
    MoWAWindowManifestEntry,
    load_mowa_window_manifest,
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

    robot_state_history: torch.Tensor
    history_actions: torch.Tensor
    action_chunk: torch.Tensor

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
        text_latent_cache_root: Path | str | None = None,
        instruction_text_latent: Path | str | MoWAInstructionTextLatentCache | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.label_sidecar_root = (
            Path(label_sidecar_root) if label_sidecar_root is not None else None
        )
        self.text_latent_cache_root = (
            Path(text_latent_cache_root)
            if text_latent_cache_root is not None
            else None
        )
        self._instruction_table: MoWAInstructionTextLatentCache | None = None
        if instruction_text_latent is not None:
            if isinstance(instruction_text_latent, MoWAInstructionTextLatentCache):
                self._instruction_table = instruction_text_latent
            else:
                self._instruction_table = MoWAInstructionTextLatentCache(
                    instruction_text_latent
                )
        self._entries = load_mowa_window_manifest(self.manifest_path)
        self._store_cache: dict[str, MoWAEpisodeLatentStore] = {}
        self._text_store_cache: dict[str, MoWATextLatentStore] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> WindowLatentSample:
        entry = self._entries[index]
        return self._assemble_sample(entry)

    def get_entry(self, index: int) -> MoWAWindowManifestEntry:
        return self._entries[index]

    def _get_store(self, path: str) -> MoWAEpisodeLatentStore:
        store = self._store_cache.get(path)
        if store is None:
            store = MoWAEpisodeLatentStore(Path(path))
            self._store_cache[path] = store
        return store

    def _get_text_store(self, episode_id: str) -> MoWATextLatentStore | None:
        if self.text_latent_cache_root is None:
            return None
        store = self._text_store_cache.get(episode_id)
        if store is None:
            store_path = self.text_latent_cache_root / f"{episode_id}.h5"
            if not store_path.is_file():
                return None
            store = MoWATextLatentStore(store_path)
            self._text_store_cache[episode_id] = store
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

        text_store = self._get_text_store(entry.episode_id)
        if text_store is None:
            return instruction
        return {
            "instruction": text_store.get_instruction(),
            "text_embeds": text_store.get_text_embeds(),
            "attention_mask": text_store.get_attention_mask(),
            "pooled_text_hidden": text_store.get_pooled_text_hidden(),
        }

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

    def _assemble_sample(self, entry: MoWAWindowManifestEntry) -> WindowLatentSample:
        validate_window_indices(entry)

        store = self._get_store(entry.episode_latent_path)

        current_latent = store.get_latents(entry.video_key, (entry.current_index,))
        history_latents = store.get_latents(entry.video_key, entry.history_indices)
        future_latents = store.get_latents(entry.video_key, entry.future_indices)

        # Squeeze the single-frame current latent if it is pooled vector form.
        if current_latent.shape[0] == 1:
            current_latent = current_latent.squeeze(0)

        robot_state = store.get_robot_state()
        robot_action = store.get_robot_action()

        robot_state_history = robot_state[list(entry.robot_state_indices), ...]
        history_actions = robot_action[list(entry.history_action_indices), ...]
        action_chunk = robot_action[list(entry.action_chunk_indices), ...]

        language = self._load_language(entry)
        labels = self._load_labels(entry)

        metadata = {
            "episode_id": entry.episode_id,
            "sample_id": entry.sample_id,
            "anchor_index": entry.anchor_index,
            "video_key": entry.video_key,
            "history_indices": entry.history_indices,
            "current_index": entry.current_index,
            "future_indices": entry.future_indices,
            "robot_state_indices": entry.robot_state_indices,
            "history_action_indices": entry.history_action_indices,
            "action_chunk_indices": entry.action_chunk_indices,
            "wam_hz": entry.wam_hz,
            "history_stride": entry.history_stride,
        }

        sample = WindowLatentSample(
            episode_id=entry.episode_id,
            sample_id=entry.sample_id,
            anchor_index=entry.anchor_index,
            current_latent=current_latent,
            history_latents=history_latents,
            future_latents=future_latents,
            robot_state_history=robot_state_history,
            history_actions=history_actions,
            action_chunk=action_chunk,
            language=language,
            labels=labels,
            metadata=metadata,
        )
        assert_no_future_leakage(sample)
        return sample



def validate_window_indices(entry: MoWAWindowManifestEntry) -> None:
    """Validate temporal boundaries of a window manifest entry."""

    if entry.current_index != entry.anchor_index:
        raise ValueError(
            f"current_index must equal anchor_index: {entry.current_index} != {entry.anchor_index}"
        )
    if not entry.history_indices:
        raise ValueError("history_indices must be non-empty.")
    if max(entry.history_indices) > entry.anchor_index:
        raise ValueError(
            f"history_indices cannot include future timesteps: max={max(entry.history_indices)}, "
            f"anchor={entry.anchor_index}"
        )
    if not entry.future_indices:
        raise ValueError("future_indices must be non-empty.")
    if min(entry.future_indices) <= entry.anchor_index:
        raise ValueError(
            f"future_indices must be strictly after anchor_index: min={min(entry.future_indices)}, "
            f"anchor={entry.anchor_index}"
        )
    if not _is_monotonic(entry.history_indices):
        raise ValueError(f"history_indices must be monotonic: {entry.history_indices}")
    if not _is_monotonic(entry.future_indices):
        raise ValueError(f"future_indices must be monotonic: {entry.future_indices}")
    if entry.action_chunk_indices and min(entry.action_chunk_indices) < entry.anchor_index:
        raise ValueError(
            f"action_chunk_indices cannot start before anchor_index: "
            f"min={min(entry.action_chunk_indices)}, anchor={entry.anchor_index}"
        )


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


def _is_monotonic(indices: tuple[int, ...]) -> bool:
    if len(indices) <= 1:
        return True
    return all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))
