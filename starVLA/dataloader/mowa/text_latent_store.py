"""MoWA episode-level text latent store builder / reader / validator.

This module caches UMT5 text encoder outputs per episode so that the
Wan2.2-TI2V world model can be trained / inferred without loading the
~4.7B-parameter text encoder.

Storage layout (one HDF5 file per episode):

    text_latent_cache/
      ep_000001.h5
      ├── /text_embeds              [1, max_length, text_hidden_dim]
      ├── /attention_mask           [1, max_length]
      ├── /pooled_text_hidden       [1, text_hidden_dim]
      ├── /instruction              raw instruction string (UTF-8 bytes)
      └── attrs:
          episode_id
          text_encoder_name
          text_encoder_version
          text_hidden_dim
          max_length
          dtype
          instruction
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import h5py
import numpy as np
import torch


_TIME_ORDER = "oldest_to_latest"


class MoWATextEncoderAdapter(Protocol):
    """Adapter for encoding an instruction into text latents."""

    encoder_name: str
    encoder_version: str
    text_hidden_dim: int
    max_length: int

    def encode_instruction(self, instruction: str) -> Mapping[str, np.ndarray]:
        """Return a dict with keys text_embeds, attention_mask, pooled_text_hidden."""


@dataclass(frozen=True)
class MoWAFakeTextEncoderAdapter:
    """Deterministic fake text encoder for unit tests and smoke checks."""

    encoder_name: str = "mowa-fake-text-encoder"
    encoder_version: str = "fake-v1"
    text_hidden_dim: int = 4096
    max_length: int = 32
    seed_salt: str = "mowa-fake-text-latent-cache"

    def encode_instruction(self, instruction: str) -> dict[str, np.ndarray]:
        payload = "|".join(
            (self.seed_salt, self.encoder_name, self.encoder_version, instruction)
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
        rng = np.random.default_rng(seed)

        attention_mask = np.ones((1, self.max_length), dtype=np.int64)
        text_embeds = rng.standard_normal(
            (1, self.max_length, self.text_hidden_dim), dtype=np.float32
        )
        pooled = (text_embeds * attention_mask[:, :, None].astype(np.float32)).sum(
            axis=1
        ) / attention_mask.sum(axis=1, keepdims=True).astype(np.float32).clip(min=1)
        return {
            "text_embeds": text_embeds,
            "attention_mask": attention_mask,
            "pooled_text_hidden": pooled,
        }


@dataclass
class MoWAUmt5TextEncoderAdapter:
    """Encode instructions with the UMT5-XXL text encoder from Wan2.2-TI2V."""

    model_path: Path | str
    encoder_name: str = "google/umt5-xxl"
    encoder_version: str = "TBD"
    text_hidden_dim: int = 4096
    max_length: int = 512
    _torch_dtype: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _text_encoder: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"UMT5 model path not found: {model_path}")

    def encode_instruction(self, instruction: str) -> dict[str, np.ndarray]:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._text_encoder is not None

        device = next(self._text_encoder.parameters()).device
        text_inputs = self._tokenizer(
            instruction,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = self._text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )
            text_embeds = outputs.last_hidden_state

        attention_mask = text_inputs.attention_mask.to(dtype=torch.int64)
        mask = attention_mask.unsqueeze(-1).to(dtype=text_embeds.dtype)
        pooled = (text_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return {
            "text_embeds": text_embeds.detach().cpu().float().numpy(),
            "attention_mask": attention_mask.detach().cpu().numpy(),
            "pooled_text_hidden": pooled.detach().cpu().float().numpy(),
        }

    def _ensure_loaded(self) -> None:
        if self._text_encoder is not None and self._tokenizer is not None:
            return
        try:
            from transformers import T5TokenizerFast, UMT5EncoderModel
        except ImportError as exc:
            raise RuntimeError("MoWA UMT5 text encoder requires transformers.") from exc

        model_path = Path(self.model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self._tokenizer = T5TokenizerFast.from_pretrained(
            str(model_path), subfolder="tokenizer"
        )
        self._text_encoder = UMT5EncoderModel.from_pretrained(
            str(model_path), subfolder="text_encoder", torch_dtype=dtype
        ).to(device)
        self._torch_dtype = dtype


@dataclass(frozen=True)
class MoWATextLatentStoreConfig:
    """Configuration for building an episode-level text latent store."""

    dataset_path: Path
    cache_root: Path
    encoder_kind: str = "fake"
    encoder_name: str = "mowa-fake-text-encoder"
    encoder_version: str = "fake-v1"
    encoder_model_path: Path | None = None
    text_encoder_name: str = "TBD"
    text_encoder_version: str = "TBD"
    text_hidden_dim: int = 4096
    max_length: int = 512
    dtype: str = "float32"
    language_source: str = "meta/tasks.jsonl"
    dry_run: bool = True
    overwrite: bool = False
    episode_indices: tuple[int, ...] | None = None

    def validate(self) -> None:
        if self.encoder_kind not in {"fake", "umt5"}:
            raise ValueError(f"encoder_kind must be fake/umt5, got {self.encoder_kind!r}.")
        if self.encoder_kind == "umt5" and self.encoder_model_path is None:
            raise ValueError("encoder_model_path must be provided when encoder_kind='umt5'.")
        if self.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError(f"Unsupported dtype: {self.dtype!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_path),
            "cache_root": str(self.cache_root),
            "encoder_kind": self.encoder_kind,
            "encoder_name": self.encoder_name,
            "encoder_version": self.encoder_version,
            "encoder_model_path": str(self.encoder_model_path) if self.encoder_model_path is not None else None,
            "text_encoder_name": self.text_encoder_name,
            "text_encoder_version": self.text_encoder_version,
            "text_hidden_dim": self.text_hidden_dim,
            "max_length": self.max_length,
            "dtype": self.dtype,
            "language_source": self.language_source,
            "dry_run": self.dry_run,
            "overwrite": self.overwrite,
            "episode_indices": self.episode_indices,
        }


@dataclass(frozen=True)
class MoWATextLatentStoreBuildReport:
    dataset_path: str
    cache_root: str
    episode_count: int
    written_count: int
    skipped_count: int
    failed_count: int
    dry_run: bool
    details: tuple[dict[str, Any], ...]
    go_no_go: str
    notes: tuple[str, ...] = (
        "Episode-level text latent store keeps UMT5 outputs bound to episodes.",
        "The world model can consume cached text latents instead of loading UMT5.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "cache_root": self.cache_root,
            "episode_count": self.episode_count,
            "written_count": self.written_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "dry_run": self.dry_run,
            "details": list(self.details),
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MoWATextLatentStoreValidationReport:
    cache_root: str
    checked_count: int
    valid_count: int
    invalid_count: int
    missing_count: int
    go_no_go: str
    notes: tuple[str, ...] = (
        "Validation checks file presence, required groups, attrs and text shape consistency.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_root": self.cache_root,
            "checked_count": self.checked_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "missing_count": self.missing_count,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


class MoWATextLatentStore:
    """Read-only accessor for a single episode text latent store HDF5 file."""

    def __init__(self, store_path: Path | str) -> None:
        self.store_path = Path(store_path)
        if not self.store_path.is_file():
            raise FileNotFoundError(f"Episode text latent store not found: {self.store_path}")

    @property
    def attrs(self) -> dict[str, Any]:
        with h5py.File(self.store_path, "r") as file:
            return dict(file.attrs)

    def get_instruction(self) -> str:
        with h5py.File(self.store_path, "r") as file:
            raw = file["/instruction"][()]
            if isinstance(raw, bytes):
                return raw.decode("utf-8")
            return str(raw)

    def get_text_embeds(self) -> torch.Tensor:
        with h5py.File(self.store_path, "r") as file:
            return torch.from_numpy(np.asarray(file["/text_embeds"][()]))

    def get_attention_mask(self) -> torch.Tensor:
        with h5py.File(self.store_path, "r") as file:
            return torch.from_numpy(np.asarray(file["/attention_mask"][()]))

    def get_pooled_text_hidden(self) -> torch.Tensor:
        with h5py.File(self.store_path, "r") as file:
            return torch.from_numpy(np.asarray(file["/pooled_text_hidden"][()]))


def build_mowa_text_latent_store(
    config: MoWATextLatentStoreConfig,
    *,
    encoder: MoWATextEncoderAdapter | None = None,
) -> MoWATextLatentStoreBuildReport:
    """Build episode-level text latent stores from a LeRobot-style dataset."""

    config.validate()
    encoder = encoder or _build_encoder_adapter(config)

    dataset_path = Path(config.dataset_path)
    cache_root = Path(config.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    from starVLA.dataloader.mowa.episode_latent_store import (
        _episode_index_from_path,
        _filter_episode_parquet_paths,
        _list_episode_parquet_paths,
        _read_instruction,
    )

    parquet_paths = _list_episode_parquet_paths(dataset_path)
    parquet_paths = _filter_episode_parquet_paths(parquet_paths, config.episode_indices)

    details: list[dict[str, Any]] = []
    written_count = 0
    skipped_count = 0
    failed_count = 0

    for parquet_path in parquet_paths:
        episode_index = _episode_index_from_path(parquet_path)
        store_path = cache_root / f"ep_{episode_index:06d}.h5"
        detail: dict[str, Any] = {"episode_index": episode_index, "store_path": str(store_path)}

        if store_path.exists() and not config.overwrite:
            skipped_count += 1
            detail["status"] = "skipped_exists"
            details.append(detail)
            continue

        instruction = _read_instruction(dataset_path, episode_index)

        if config.dry_run:
            detail["status"] = "planned"
            detail["instruction"] = instruction
            details.append(detail)
            continue

        try:
            encoded = encoder.encode_instruction(instruction)
            tmp_path = store_path.with_suffix(".h5.tmp")
            with h5py.File(tmp_path, "w") as file:
                _write_text_metadata(file, config, episode_index, instruction)
                file.create_dataset("/text_embeds", data=encoded["text_embeds"], compression="gzip")
                file.create_dataset("/attention_mask", data=encoded["attention_mask"], compression="gzip")
                file.create_dataset(
                    "/pooled_text_hidden", data=encoded["pooled_text_hidden"], compression="gzip"
                )
                file.create_dataset("/instruction", data=instruction.encode("utf-8"))

            tmp_path.replace(store_path)
            written_count += 1
            detail["status"] = "written"
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            detail["status"] = "failed"
            detail["error"] = str(exc)
            if store_path.exists():
                store_path.unlink()

        details.append(detail)

    go_no_go = (
        "TBD: episode text latent store write completed; validate before integration"
        if written_count > 0
        else (
            "TBD: episode text latent store planned; execute and validate before integration"
            if not config.dry_run
            else "No-Go: no episode text latent stores were planned"
        )
    )
    if failed_count > 0:
        go_no_go = "No-Go: some episode text latent stores failed to build"

    return MoWATextLatentStoreBuildReport(
        dataset_path=str(dataset_path),
        cache_root=str(cache_root),
        episode_count=len(parquet_paths),
        written_count=written_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        dry_run=config.dry_run,
        details=tuple(details),
        go_no_go=go_no_go,
    )


def validate_mowa_text_latent_store(
    cache_root: Path | str,
) -> MoWATextLatentStoreValidationReport:
    """Validate all episode text latent stores under ``cache_root``."""

    cache_root = Path(cache_root)
    store_paths = sorted(cache_root.glob("ep_*.h5"))

    checked_count = 0
    valid_count = 0
    invalid_count = 0
    missing_count = 0

    for store_path in store_paths:
        checked_count += 1
        try:
            store = MoWATextLatentStore(store_path)
            attrs = store.attrs
            required_attrs = {
                "episode_id",
                "text_encoder_name",
                "text_hidden_dim",
                "max_length",
                "dtype",
            }
            missing_attrs = required_attrs - set(attrs.keys())
            if missing_attrs:
                invalid_count += 1
                continue

            text_embeds = store.get_text_embeds()
            attention_mask = store.get_attention_mask()
            pooled = store.get_pooled_text_hidden()
            if text_embeds.dim() != 3 or attention_mask.dim() != 2 or pooled.dim() != 2:
                invalid_count += 1
                continue
            if (
                text_embeds.shape[1] != int(attrs["max_length"])
                or text_embeds.shape[2] != int(attrs["text_hidden_dim"])
            ):
                invalid_count += 1
                continue
            if attention_mask.shape != text_embeds.shape[:2]:
                invalid_count += 1
                continue
            if pooled.shape != (text_embeds.shape[0], text_embeds.shape[2]):
                invalid_count += 1
                continue
            valid_count += 1
        except FileNotFoundError:
            missing_count += 1
        except Exception:  # noqa: BLE001
            invalid_count += 1

    all_ok = checked_count > 0 and invalid_count == 0 and missing_count == 0
    return MoWATextLatentStoreValidationReport(
        cache_root=str(cache_root),
        checked_count=checked_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        missing_count=missing_count,
        go_no_go=(
            "TBD: episode text latent store validation passed"
            if all_ok
            else "No-Go: episode text latent store validation failed"
        ),
    )


def _build_encoder_adapter(config: MoWATextLatentStoreConfig) -> MoWATextEncoderAdapter:
    if config.encoder_kind == "umt5":
        if config.encoder_model_path is None:
            raise ValueError("encoder_model_path is required for UMT5 encoder.")
        return MoWAUmt5TextEncoderAdapter(
            model_path=config.encoder_model_path,
            encoder_name=config.encoder_name,
            encoder_version=config.encoder_version,
            text_hidden_dim=config.text_hidden_dim,
            max_length=config.max_length,
        )
    if config.encoder_kind == "fake":
        return MoWAFakeTextEncoderAdapter(
            encoder_name=config.encoder_name,
            encoder_version=config.encoder_version,
            text_hidden_dim=config.text_hidden_dim,
            max_length=config.max_length,
        )
    raise NotImplementedError(f"Encoder kind {config.encoder_kind!r} not yet supported.")


def _write_text_metadata(
    file: h5py.File,
    config: MoWATextLatentStoreConfig,
    episode_index: int,
    instruction: str,
) -> None:
    file.attrs["episode_id"] = f"ep_{episode_index:06d}"
    file.attrs["episode_index"] = episode_index
    file.attrs["text_encoder_name"] = config.text_encoder_name
    file.attrs["text_encoder_version"] = config.text_encoder_version
    file.attrs["text_hidden_dim"] = config.text_hidden_dim
    file.attrs["max_length"] = config.max_length
    file.attrs["dtype"] = config.dtype
    file.attrs["encoder_kind"] = config.encoder_kind
    file.attrs["encoder_name"] = config.encoder_name
    file.attrs["encoder_version"] = config.encoder_version
    file.attrs["encoder_model_path"] = (
        str(config.encoder_model_path) if config.encoder_model_path is not None else ""
    )
    file.attrs["instruction"] = instruction
