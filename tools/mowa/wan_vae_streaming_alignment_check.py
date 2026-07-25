"""核对在线流式 Wan VAE 与训练 episode cache 的 regular latent。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from starVLA.dataloader.mowa.episode_latent_store import (
    MoWAWanVaeEpisodeEncoderAdapter,
    _resolve_video_path,
)
from starVLA.model.modules.world_model.wan_vae_utils import WanVAEStreamEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-store", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--num-regular-latents", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with h5py.File(args.episode_store, "r") as store:
        dataset_path = Path(str(store.attrs["dataset_path"]))
        episode_index = int(store.attrs["episode_index"])
        cached = np.asarray(store[f"latents/{args.video_key}"][: args.num_regular_latents + 1], dtype=np.float32)
        source_indices = np.asarray(
            store[f"indices/latent_source_frame_indices/{args.video_key}"][
                : args.num_regular_latents + 1
            ],
            dtype=np.int64,
        )

    if len(cached) < args.num_regular_latents + 1:
        raise ValueError(
            f"Episode store only has {len(cached) - 1} regular latents; "
            f"requested {args.num_regular_latents}."
        )
    video_path = _resolve_video_path(dataset_path, args.video_key, episode_index)
    endpoint = int(source_indices[args.num_regular_latents])
    adapter = MoWAWanVaeEpisodeEncoderAdapter(
        model_path=args.model_path,
        video_backend="pyav",
    )
    frames = adapter.load_video_frames(video_path, tuple(range(endpoint + 1)))
    adapter._ensure_loaded()
    assert adapter._vae is not None and adapter._video_processor is not None
    stream = WanVAEStreamEncoder(adapter._vae, adapter._video_processor)

    stream.encode("alignment", [frames[0]], reset=True, max_regular_latents=args.num_regular_latents)
    streamed = []
    for regular_index in range(args.num_regular_latents):
        start = 1 + regular_index * 4
        latents = stream.encode(
            "alignment",
            frames[start : start + 4],
            max_regular_latents=args.num_regular_latents,
        )
        streamed.append(latents[-1].squeeze(0).squeeze(1).float().cpu().numpy())
    streamed_array = np.stack(streamed)
    cached_regular = cached[1 : args.num_regular_latents + 1]
    difference = streamed_array - cached_regular
    per_step_mse = np.mean(np.square(difference), axis=(1, 2, 3))
    per_step_max_abs = np.max(np.abs(difference), axis=(1, 2, 3))
    per_step_cosine = [
        float(
            np.dot(streamed_array[index].reshape(-1), cached_regular[index].reshape(-1))
            / (
                np.linalg.norm(streamed_array[index])
                * np.linalg.norm(cached_regular[index])
                + 1e-12
            )
        )
        for index in range(args.num_regular_latents)
    ]
    report = {
        "episode_store": str(args.episode_store),
        "video_path": str(video_path),
        "video_key": args.video_key,
        "source_endpoints": source_indices[1 : args.num_regular_latents + 1].tolist(),
        "shape": list(streamed_array.shape),
        "per_step_mse": per_step_mse.tolist(),
        "per_step_max_abs": per_step_max_abs.tolist(),
        "per_step_cosine": per_step_cosine,
        "max_mse": float(per_step_mse.max()),
        "min_cosine": float(min(per_step_cosine)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
