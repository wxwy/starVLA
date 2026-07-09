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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from starVLA.dataloader.mowa.text_latent_store import MoWAUmt5TextEncoderAdapter


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
    for instruction in instructions:
        try:
            encoded = encoder.encode_instruction(instruction)
            table[instruction] = {
                "text_embeds": torch.from_numpy(encoded["text_embeds"]),
                "attention_mask": torch.from_numpy(encoded["attention_mask"]),
                "pooled_text_hidden": torch.from_numpy(encoded["pooled_text_hidden"]),
            }
        except Exception:  # noqa: BLE001
            failed_count += 1

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
