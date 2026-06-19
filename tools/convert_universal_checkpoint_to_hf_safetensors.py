#!/usr/bin/env python
"""Convert a DeepSpeed Universal checkpoint to HF safetensors model shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.training.checkpoints import convert_universal_checkpoint_to_hf_safetensors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Universal checkpoint model weights to lightweight/HF safetensors shards"
    )
    parser.add_argument("--input_folder", required=True, help="Universal checkpoint tag folder or mp_rank_00_model_states.pt")
    parser.add_argument("--output_folder", required=True, help="HF safetensors output folder")
    parser.add_argument("--max_shard_size", default="5GB", help="Shard size, e.g. 4GB or 500MB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_folder = convert_universal_checkpoint_to_hf_safetensors(
        args.input_folder,
        args.output_folder,
        max_shard_size=args.max_shard_size,
        overwrite=args.overwrite,
    )
    print(f"HF safetensors model shards saved at: {output_folder}", flush=True)


if __name__ == "__main__":
    main()
