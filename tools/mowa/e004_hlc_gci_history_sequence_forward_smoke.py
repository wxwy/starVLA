"""E-004 HLC-GCI history sequence forward smoke.

Validates that the latent cache now stores a real per-step history sequence
and that HLC-GCI can consume it without the runtime repeat bridge.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from starVLA.dataloader.mowa.latent_cache_builder import (
    MoWALatentCacheBuildConfig,
    MoWALatentCacheDataset,
    build_mowa_latent_cache,
)
from starVLA.model.modules.mowa import (
    MoWAHLCGCI,
    MoWAHLCGCIConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test E-004 HLC-GCI with a real history sequence cache."
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _write_minimal_episode(dataset_path: Path, *, length: int = 8) -> None:
    data_dir = dataset_path / "data" / "chunk-000"
    video_dir = (
        dataset_path
        / "videos"
        / "chunk-000"
        / "observation.images.robot0_agentview_left"
    )
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)

    parquet = pa.table(
        {
            "frame_index": pa.array(list(range(length)), type=pa.int64()),
            "episode_index": pa.array([0] * length, type=pa.int64()),
            "task_index": pa.array([0] * length, type=pa.int64()),
            "timestamp": pa.array([0.05 * idx for idx in range(length)], type=pa.float32()),
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array([float(value) for value in range(length * 16)]),
                16,
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array([float(value) for value in range(length * 12)]),
                12,
            ),
            "next.reward": pa.array([0.0] * (length - 1) + [1.0], type=pa.float32()),
            "next.done": pa.array([False] * (length - 1) + [True]),
        }
    )
    pq.write_table(parquet, data_dir / "episode_000000.parquet")
    (video_dir / "episode_000000.mp4").write_bytes(b"fake-video-bytes")


def main() -> None:
    args = parse_args()
    payload = build_e004_hlc_gci_history_sequence_forward_smoke()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e004_hlc_gci_history_sequence_forward_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        dataset_path = root / "dataset"
        cache_root = root / "cache"
        _write_minimal_episode(dataset_path, length=8)

        history_window_steps = 3
        latent_dim = 16

        build_report = build_mowa_latent_cache(
            MoWALatentCacheBuildConfig(
                dataset_path=dataset_path,
                cache_root=cache_root,
                encoder_name="wan-fake-encoder",
                encoder_version="fake-v1",
                latent_dim=latent_dim,
                video_keys=("observation.images.robot0_agentview_left",),
                current_window_steps=1,
                future_window_steps=2,
                history_window_steps=history_window_steps,
                anchor_mode="smoke",
                dry_run=False,
            )
        ).to_dict()

        dataset = MoWALatentCacheDataset(cache_root)
        sample = dataset.get_sample(
            episode_index=0,
            anchor_index=2,
            video_key="observation.images.robot0_agentview_left",
        )

        history_latent = sample["history_latent"]
        cache_shape_ok = (
            isinstance(history_latent, torch.Tensor)
            and tuple(history_latent.shape) == (history_window_steps, latent_dim)
        )
        history_steps_differ = False
        if cache_shape_ok:
            history_steps_differ = not torch.allclose(history_latent[0], history_latent[1])

        hlc_gci = MoWAHLCGCI(
            MoWAHLCGCIConfig(
                history_latent_dim=latent_dim,
                condition_hidden_dim=latent_dim,
                history_steps=history_window_steps,
                compressed_history_dim=8,
                gate_hidden_dim=8,
            )
        )
        batch_history = history_latent.unsqueeze(0)  # [1, history_steps, latent_dim]
        condition_tokens = torch.randn(1, 4, latent_dim)
        with torch.inference_mode():
            hlc_output = hlc_gci(batch_history, condition_tokens)

        gate_in_range = bool(
            torch.all(hlc_output.gate_values >= 0.0) and torch.all(hlc_output.gate_values <= 1.0)
        )
        output_shapes_ok = (
            tuple(hlc_output.compressed_history.shape) == (1, 8)
            and tuple(hlc_output.gate_values.shape) == (1, latent_dim)
            and tuple(hlc_output.gated_condition_tokens.shape) == (1, 4, latent_dim)
        )

    checks = {
        "cache_built": build_report["written_artifact_count"] == 3,
        "history_latent_is_2d_sequence": cache_shape_ok,
        "history_timesteps_are_distinct": history_steps_differ,
        "hlc_gci_output_shapes_ok": output_shapes_ok,
        "hlc_gci_gate_values_in_0_1": gate_in_range,
    }

    return {
        "stage": "hlc_gci",
        "experiment_id": "E-004",
        "config_role": "history_sequence_forward_smoke",
        "checks": checks,
        "observed": {
            "history_latent_shape": tuple(history_latent.shape) if isinstance(history_latent, torch.Tensor) else None,
            "compressed_history_shape": tuple(hlc_output.compressed_history.shape),
            "gate_values_shape": tuple(hlc_output.gate_values.shape),
            "gated_condition_tokens_shape": tuple(hlc_output.gated_condition_tokens.shape),
        },
        "go_no_go": (
            "TBD: E-004 history sequence cache contract validated; HLC-GCI forward consumes real sequence"
            if all(checks.values())
            else "No-Go: E-004 history sequence cache contract or HLC-GCI forward failed"
        ),
        "unresolved_items": [
            "This smoke only validates cache shape and a standalone HLC-GCI forward pass.",
            "Next step is a QwenPI_v3 forward smoke with HLC-GCI enabled and a real history sequence batch.",
        ],
    }


if __name__ == "__main__":
    main()
