"""CLI to build an instruction-level UMT5 text latent cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa.instruction_text_latent_cache import (
    build_mowa_instruction_text_latent_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single instruction-to-latent table for all target atomic tasks."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365/v1.0/target/atomic"),
        help="Root directory containing task subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .pt path. If omitted, the table is kept in memory only.",
    )
    parser.add_argument(
        "--encoder-model-path",
        type=Path,
        default=Path("playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        help="Local Wan2.2-TI2V-5B-Diffusers directory.",
    )
    parser.add_argument(
        "--text-hidden-dim",
        type=int,
        default=4096,
        help="UMT5 hidden dimension.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max_length.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float16", "float32", "bfloat16"),
        help="Stored latent dtype.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, _cache = build_mowa_instruction_text_latent_cache(
        dataset_root=args.dataset_root,
        output_path=args.output,
        encoder_model_path=args.encoder_model_path,
        text_hidden_dim=args.text_hidden_dim,
        max_length=args.max_length,
        dtype=args.dtype,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
