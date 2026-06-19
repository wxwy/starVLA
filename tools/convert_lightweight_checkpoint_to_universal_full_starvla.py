#!/usr/bin/env python
"""Convert a StarVLA lightweight ZeRO checkpoint to full-adam Universal Checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.training.checkpoints.universal import (
    _load_model_shapes,
    convert_lightweight_checkpoint_to_universal,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert StarVLA lightweight ZeRO checkpoint to full DeepSpeed Universal Checkpoint"
    )
    parser.add_argument("--input_folder", required=True, help="StarVLA lightweight checkpoint tag folder")
    parser.add_argument("--output_folder", required=True, help="Universal checkpoint tag output folder")
    parser.add_argument(
        "--model_state_file",
        required=True,
        help="Native DeepSpeed mp_rank_00_model_states.pt file matching the checkpoint model",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder = Path(args.input_folder).resolve()
    output_folder = Path(args.output_folder).resolve()
    model_state_file = Path(args.model_state_file).resolve()

    try:
        shapes = _load_model_shapes(input_folder)
        print(f"Model tensors in safetensors: {len(shapes)}", flush=True)
    except FileNotFoundError:
        shapes = None

    convert_lightweight_checkpoint_to_universal(
        input_folder,
        output_folder,
        model_state_file,
        overwrite=args.overwrite,
        shapes=shapes,
    )


if __name__ == "__main__":
    main()
