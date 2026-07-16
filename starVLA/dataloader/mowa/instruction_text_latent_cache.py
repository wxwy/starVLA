"""MoWA instruction-level text latent cache.

Instead of storing a per-episode text latent HDF5, this module builds a single
lookup table that maps each unique language instruction to its UMT5 outputs:

    text_embeds        [1, max_length, text_hidden_dim]
    attention_mask     [1, max_length]
    pooled_text_hidden [1, text_hidden_dim]

Because MoWA target atomic tasks have only a few hundred unique instructions,
this table is small (~1-2 GB in fp32) and can be kept in CPU memory.  Training
and inference can then drop the ~4.7B-parameter UMT5 text encoder entirely.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


@dataclass
class MoWAUmt5TextEncoderAdapter:
    """Encode instructions with the UMT5-XXL text encoder from Wan2.2-TI2V."""

    model_path: Path | str
    encoder_name: str = "google/umt5-xxl"
    encoder_version: str = "TBD"
    text_hidden_dim: int = 4096
    max_length: int = 512
    _tokenizer: Any = None
    _text_encoder: Any = None

    def __post_init__(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"UMT5 model path not found: {self.model_path}")

    def encode_instruction(self, instruction: str) -> dict[str, torch.Tensor]:
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
            text_embeds = self._text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state
        attention_mask = text_inputs.attention_mask.to(dtype=torch.int64)
        mask = attention_mask.unsqueeze(-1).to(dtype=text_embeds.dtype)
        pooled = (text_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return {
            "text_embeds": text_embeds.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "pooled_text_hidden": pooled.detach().cpu(),
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
        print(
            f"[instruction-text-cache] loading UMT5 model={model_path} device={device} "
            f"compute_dtype={dtype}",
            flush=True,
        )
        self._tokenizer = T5TokenizerFast.from_pretrained(str(model_path), subfolder="tokenizer")
        self._text_encoder = UMT5EncoderModel.from_pretrained(
            str(model_path), subfolder="text_encoder", torch_dtype=dtype
        ).to(device)
        print("[instruction-text-cache] UMT5 ready", flush=True)


@dataclass(frozen=True)
class MoWAInstructionTextLatentCacheBuildReport:
    output_path: str | None
    instruction_count: int
    encoded_count: int
    failed_count: int
    text_hidden_dim: int
    max_length: int
    dtype: str
    go_no_go: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "instruction_count": self.instruction_count,
            "encoded_count": self.encoded_count,
            "failed_count": self.failed_count,
            "text_hidden_dim": self.text_hidden_dim,
            "max_length": self.max_length,
            "dtype": self.dtype,
            "go_no_go": self.go_no_go,
        }


class MoWAInstructionTextLatentCache:
    """In-memory lookup table from instruction string to UMT5 latents."""

    def __init__(self, cache_path: Path | str) -> None:
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Instruction text latent cache not found: {self.cache_path}")
        data = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self._table: dict[str, dict[str, torch.Tensor]] = data["instruction_to_latents"]
        self.metadata: dict[str, Any] = data.get("metadata", {})

    @classmethod
    def from_table(
        cls,
        table: dict[str, dict[str, torch.Tensor]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "MoWAInstructionTextLatentCache":
        """Create an in-memory cache directly from a built table."""
        instance = cls.__new__(cls)
        instance.cache_path = Path("memory-only")
        instance._table = table
        instance.metadata = metadata or {}
        return instance

    def save(self, output_path: Path | str) -> None:
        """Persist the in-memory table to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "instruction_to_latents": self._table,
                "metadata": self.metadata,
            },
            output_path,
        )
        self.cache_path = output_path

    def lookup(self, instruction: str) -> dict[str, torch.Tensor] | None:
        return self._table.get(instruction)

    def __len__(self) -> int:
        return len(self._table)


def _collect_unique_instructions(dataset_root: Path) -> list[str]:
    """Collect unique instruction strings from all LeRobot episodes under root."""
    unique: set[str] = set()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    for task_dir in sorted(dataset_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for date_dir in sorted(task_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            episodes_path = date_dir / "lerobot" / "meta" / "episodes.jsonl"
            if not episodes_path.is_file():
                continue
            with episodes_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    tasks = data.get("tasks")
                    if isinstance(tasks, list):
                        for task in tasks:
                            if task:
                                unique.add(str(task))
    return sorted(unique)



def build_mowa_instruction_text_latent_cache(
    dataset_root: Path | str,
    output_path: Path | str | None = None,
    *,
    encoder_model_path: Path | str,
    text_hidden_dim: int = 4096,
    max_length: int = 512,
    dtype: str = "float32",
) -> tuple[MoWAInstructionTextLatentCacheBuildReport, MoWAInstructionTextLatentCache]:
    """Build a single instruction-to-latent table for all tasks under ``dataset_root``.

    Args:
        dataset_root: Root directory containing task subdirectories.
        output_path: If provided, the table is persisted to this path. If ``None``,
            the table is kept in memory only.
        encoder_model_path: Local Wan2.2-TI2V-5B-Diffusers directory (contains UMT5).
        text_hidden_dim: UMT5 hidden dimension.
        max_length: Tokenizer max_length.
        dtype: Stored latent dtype.

    Returns:
        A tuple of (build_report, in_memory_cache). The cache is always returned
        so callers can use it immediately even when ``output_path`` is ``None``.
    """

    dataset_root = Path(dataset_root)

    instructions = _collect_unique_instructions(dataset_root)
    if not instructions:
        raise ValueError(f"No instructions found under {dataset_root}")

    encoder = MoWAUmt5TextEncoderAdapter(
        model_path=encoder_model_path,
        encoder_name="google/umt5-xxl",
        encoder_version="TBD",
        text_hidden_dim=text_hidden_dim,
        max_length=max_length,
    )

    table: dict[str, dict[str, torch.Tensor]] = {}
    failed_count = 0
    storage_dtype = getattr(torch, dtype)
    print("[instruction-text-cache] configuration", flush=True)
    print(f"  dataset_root: {dataset_root}", flush=True)
    print(f"  output_path: {output_path or 'memory-only'}", flush=True)
    print(f"  model_path: {encoder_model_path}", flush=True)
    print(
        f"  instructions: {len(instructions)} | storage_dtype: {dtype} | "
        f"max_length: {max_length}",
        flush=True,
    )
    started_at = time.monotonic()
    progress_bar = tqdm(
        total=len(instructions),
        desc="Encoding text",
        unit="instruction",
        dynamic_ncols=True,
    )
    for instruction_index, instruction in enumerate(instructions):
        try:
            encode_started_at = time.monotonic()
            encoded = encoder.encode_instruction(instruction)
            table[instruction] = {
                "text_embeds": encoded["text_embeds"].to(dtype=storage_dtype),
                "attention_mask": encoded["attention_mask"],
                "pooled_text_hidden": encoded["pooled_text_hidden"].to(dtype=storage_dtype),
            }
            token_count = int(encoded["attention_mask"].sum())
            encode_seconds = time.monotonic() - encode_started_at
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            progress_bar.write(
                f"[instruction-text-cache] failed instruction={instruction_index}: {exc}"
            )
            token_count = "-"
            encode_seconds = 0.0
        progress_bar.set_postfix(
            instruction=instruction_index + 1,
            tokens=token_count,
            text=f"{encode_seconds:.1f}s",
            failed=failed_count,
            refresh=False,
        )
        progress_bar.update(1)

    progress_bar.close()

    metadata: dict[str, Any] = {
        "text_encoder_name": "google/umt5-xxl",
        "text_encoder_version": "TBD",
        "text_hidden_dim": text_hidden_dim,
        "max_length": max_length,
        "dtype": dtype,
        "instruction_count": len(instructions),
        "encoded_count": len(table),
        "failed_count": failed_count,
    }
    cache = MoWAInstructionTextLatentCache.from_table(table, metadata=metadata)

    if output_path is not None:
        output_path = Path(output_path)
        cache.save(output_path)

    print(
        f"[instruction-text-cache] finished encoded={len(table)} failed={failed_count} "
        f"elapsed={(time.monotonic() - started_at) / 60:.1f} min",
        flush=True,
    )

    all_ok = len(table) == len(instructions)
    report = MoWAInstructionTextLatentCacheBuildReport(
        output_path=str(output_path) if output_path is not None else None,
        instruction_count=len(instructions),
        encoded_count=len(table),
        failed_count=failed_count,
        text_hidden_dim=text_hidden_dim,
        max_length=max_length,
        dtype=dtype,
        go_no_go=(
            "TBD: instruction text latent cache built"
            if all_ok
            else "No-Go: some instructions failed to encode"
        ),
    )
    return report, cache
