#!/usr/bin/env python
"""Convert a StarVLA DeepSpeed ZeRO checkpoint to Universal Checkpoint.

DeepSpeed's stock ds_to_universal assumes every tensor in param_shapes has
optimizer fragments. StarVLA commonly freezes large backbones, so those tensors
exist in model state but not in ZeRO optimizer param_slice_mappings. This tool
keeps DeepSpeed's conversion logic but merges only optimizer-backed tensors.
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

torch = None
DeepSpeedCheckpoint = None
PARAM_SHAPES = None
PARAM_SLICE_MAPPINGS = None
UNIVERSAL_CHECKPOINT_INFO = None
UNIVERSAL_CHECKPOINT_VERSION_KEY = None
UNIVERSAL_CHECKPOINT_VERSION_VALUE = None
_create_checkpoint_paths = None
_extract_zero_shard_files = None
_get_optim_files = None
_get_zero_stage = None
_merge_tp_slice_files = None
_save_optimizer_state = None
PARAM = "param"
CAT_DIM = "cat_dim"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a StarVLA ZeRO checkpoint into DeepSpeed Universal Checkpoint"
    )
    parser.add_argument("--input_folder", required=True, help="DeepSpeed ZeRO tag folder")
    parser.add_argument("--output_folder", required=True, help="Universal checkpoint tag folder")
    parser.add_argument("--num_extract_workers", type=int, default=4)
    parser.add_argument("--num_merge_workers", type=int, default=2)
    parser.add_argument("--keep_temp_folder", action="store_true")
    parser.add_argument("--no_strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    return parser.parse_args()


def _load_deepspeed_helpers():
    global torch
    global DeepSpeedCheckpoint
    global PARAM_SHAPES
    global PARAM_SLICE_MAPPINGS
    global UNIVERSAL_CHECKPOINT_INFO
    global UNIVERSAL_CHECKPOINT_VERSION_KEY
    global UNIVERSAL_CHECKPOINT_VERSION_VALUE
    global _create_checkpoint_paths
    global _extract_zero_shard_files
    global _get_optim_files
    global _get_zero_stage
    global _merge_tp_slice_files
    global _save_optimizer_state

    import torch as torch_module
    from deepspeed.checkpoint import (
        PARAM_SHAPES as PARAM_SHAPES_VALUE,
        PARAM_SLICE_MAPPINGS as PARAM_SLICE_MAPPINGS_VALUE,
        UNIVERSAL_CHECKPOINT_INFO as UNIVERSAL_CHECKPOINT_INFO_VALUE,
        UNIVERSAL_CHECKPOINT_VERSION_KEY as UNIVERSAL_CHECKPOINT_VERSION_KEY_VALUE,
        UNIVERSAL_CHECKPOINT_VERSION_VALUE as UNIVERSAL_CHECKPOINT_VERSION_VALUE_VALUE,
    )
    from deepspeed.checkpoint import DeepSpeedCheckpoint as DeepSpeedCheckpointValue
    from deepspeed.checkpoint.ds_to_universal import (
        _create_checkpoint_paths as _create_checkpoint_paths_value,
        _extract_zero_shard_files as _extract_zero_shard_files_value,
        _get_optim_files as _get_optim_files_value,
        _get_zero_stage as _get_zero_stage_value,
        _merge_tp_slice_files as _merge_tp_slice_files_value,
        _save_optimizer_state as _save_optimizer_state_value,
    )

    torch = torch_module
    DeepSpeedCheckpoint = DeepSpeedCheckpointValue
    PARAM_SHAPES = PARAM_SHAPES_VALUE
    PARAM_SLICE_MAPPINGS = PARAM_SLICE_MAPPINGS_VALUE
    UNIVERSAL_CHECKPOINT_INFO = UNIVERSAL_CHECKPOINT_INFO_VALUE
    UNIVERSAL_CHECKPOINT_VERSION_KEY = UNIVERSAL_CHECKPOINT_VERSION_KEY_VALUE
    UNIVERSAL_CHECKPOINT_VERSION_VALUE = UNIVERSAL_CHECKPOINT_VERSION_VALUE_VALUE
    _create_checkpoint_paths = _create_checkpoint_paths_value
    _extract_zero_shard_files = _extract_zero_shard_files_value
    _get_optim_files = _get_optim_files_value
    _get_zero_stage = _get_zero_stage_value
    _merge_tp_slice_files = _merge_tp_slice_files_value
    _save_optimizer_state = _save_optimizer_state_value


def _ensure_universal_info(ds_checkpoint):
    info = ds_checkpoint.get_checkpoint_info(UNIVERSAL_CHECKPOINT_INFO)
    if info is None:
        info = {}
    info.setdefault(UNIVERSAL_CHECKPOINT_VERSION_KEY, UNIVERSAL_CHECKPOINT_VERSION_VALUE)
    ds_checkpoint.global_state[UNIVERSAL_CHECKPOINT_INFO] = info


def _optimizer_param_names(ds_checkpoint) -> set[str]:
    names = set()
    for pp_index in range(ds_checkpoint.pp_degree):
        for tp_index in range(ds_checkpoint.tp_degree):
            for dp_index in range(ds_checkpoint.dp_degree):
                state = ds_checkpoint.get_zero_checkpoint_state(
                    pp_index=pp_index,
                    tp_index=tp_index,
                    dp_index=dp_index,
                )
                optim_state = state["optimizer_state_dict"]
                for group_mapping in optim_state[PARAM_SLICE_MAPPINGS]:
                    names.update(group_mapping.keys())
    return names


def _optimizer_param_shapes(ds_checkpoint, optimizer_names: set[str]) -> dict:
    shapes = {}
    for mp_rank_file in ds_checkpoint.mp_rank_files:
        mp_state = torch.load(mp_rank_file, map_location="cpu", weights_only=False, mmap=True)
        for shape_group in mp_state[PARAM_SHAPES]:
            for name, shape in shape_group.items():
                if name in optimizer_names:
                    shapes[name] = shape
        del mp_state
    return shapes


def _copy_model_state_files(input_folder: Path, output_folder: Path):
    for file_path in glob.glob(str(input_folder / "mp*")):
        shutil.copy2(file_path, output_folder)


def _write_latest_universal(output_folder: Path):
    checkpoint_root_folder = output_folder.parent
    latest_file = checkpoint_root_folder / "latest_universal"
    latest_file.write_text(output_folder.name, encoding="utf-8")


def _preview_files(path: Path, limit: int = 10) -> list[str]:
    if not path.exists():
        return []
    return [str(file_path) for file_path in list(path.rglob("*"))[:limit]]


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for file_path in path.rglob("*") if file_path.is_file())


def _dp_index_to_str(dp_index: int) -> str:
    return f"{dp_index:0>2d}"


def _save_torch(file_path: Path, payload):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, file_path)


def _dump_fp32_fragments(input_folder: Path, temp_dir: Path):
    optim_files = sorted(input_folder.glob("*_optim_states.pt"))
    if not optim_files:
        raise FileNotFoundError(f"No *_optim_states.pt files found in {input_folder}")

    for dp_index, optim_file in enumerate(optim_files):
        state = torch.load(optim_file, map_location="cpu", weights_only=False, mmap=True)
        optim_state = state["optimizer_state_dict"]
        fp32_groups = optim_state["single_partition_of_fp32_groups"]
        param_slice_mappings = optim_state["param_slice_mappings"]

        for group_index, group_mapping in enumerate(param_slice_mappings):
            flat_fp32 = fp32_groups[group_index]
            for name, fragment_mapping in group_mapping.items():
                fragment = flat_fp32.narrow(0, fragment_mapping.start, fragment_mapping.numel).clone()
                _save_torch(temp_dir / name / "0" / f"fp32.{_dp_index_to_str(dp_index)}", fragment)

        del state


def _merge_fp32_fragments(output_folder: Path, temp_dir: Path, slice_shapes: dict):
    zero_output_folder = output_folder / "zero"
    for name, shape in slice_shapes.items():
        prefix = temp_dir / name / "0" / "fp32"
        paths = sorted(prefix.parent.glob("fp32.*"))
        if not paths:
            raise RuntimeError(f"No fp32 fragments found for parameter: {name}")

        shards = [torch.load(path, weights_only=False) for path in paths]
        param = torch.cat(shards, dim=0).reshape(shape)
        _save_torch(
            zero_output_folder / name / "fp32.pt",
            {
                PARAM: param,
                CAT_DIM: 0,
            },
        )


def main():
    args = parse_args()
    print("Loading DeepSpeed conversion helpers...", flush=True)
    _load_deepspeed_helpers()

    input_folder = Path(args.input_folder).resolve()
    output_folder = Path(args.output_folder).resolve()

    print(f"Reading optimizer metadata from: {input_folder}", flush=True)
    optim_files = _get_optim_files(str(input_folder))
    zero_stage = _get_zero_stage(optim_files)
    if zero_stage > 2:
        raise NotImplementedError("StarVLA helper currently supports ZeRO stage 1/2 checkpoints only.")

    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Converting StarVLA ZeRO checkpoint: {input_folder}", flush=True)
    print(f"Universal output folder: {output_folder}", flush=True)

    print("Building DeepSpeed checkpoint index...", flush=True)
    ds_checkpoint = DeepSpeedCheckpoint(str(input_folder))
    _ensure_universal_info(ds_checkpoint)

    print("Collecting optimizer-backed parameter names...", flush=True)
    optimizer_names = _optimizer_param_names(ds_checkpoint)
    print(f"Optimizer-backed parameter names: {len(optimizer_names)}", flush=True)
    print("Collecting optimizer-backed parameter shapes...", flush=True)
    slice_shapes = _optimizer_param_shapes(ds_checkpoint, optimizer_names)
    missing_shapes = optimizer_names.difference(slice_shapes)
    if missing_shapes:
        preview = sorted(missing_shapes)[:20]
        raise RuntimeError(f"Optimizer params missing from param_shapes: {preview}")

    skipped_count = 0
    for mp_rank_file in ds_checkpoint.mp_rank_files:
        mp_state = torch.load(mp_rank_file, map_location="cpu", weights_only=False, mmap=True)
        total_count = sum(len(shape_group) for shape_group in mp_state[PARAM_SHAPES])
        skipped_count += total_count - len(slice_shapes)
        del mp_state

    print(f"Optimizer-backed tensors: {len(slice_shapes)}", flush=True)
    print(f"Skipped non-optimizer tensors: {skipped_count}", flush=True)

    iteration = ds_checkpoint.get_iteration()
    _create_checkpoint_paths(str(output_folder), iteration, ds_checkpoint.tp_degree, ds_checkpoint.pp_degree)
    temp_dir = str(output_folder / "tmp")

    print("*** 1. Extracting ZeRO fp32 fragments", flush=True)
    _dump_fp32_fragments(input_folder, Path(temp_dir))
    temp_path = Path(temp_dir)
    fragment_count = _count_files(temp_path)
    print(f"Extracted fragment files: {fragment_count}", flush=True)
    print(f"Fragment preview: {_preview_files(temp_path)}", flush=True)
    if fragment_count == 0:
        raise RuntimeError(f"No ZeRO fragment files were extracted under {temp_path}")

    print("*** 2. Merging optimizer-backed fp32 slices", flush=True)
    _merge_fp32_fragments(output_folder, temp_path, slice_shapes)

    print("*** 3. Saving common optimizer states", flush=True)
    _save_optimizer_state(args, ds_checkpoint)

    if not args.keep_temp_folder:
        shutil.rmtree(temp_dir, ignore_errors=True)

    _copy_model_state_files(input_folder, output_folder)
    _write_latest_universal(output_folder)

    print("*** Done", flush=True)


if __name__ == "__main__":
    main()
