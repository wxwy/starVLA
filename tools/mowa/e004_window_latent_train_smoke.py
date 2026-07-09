"""E-004 window-latent training smoke for MoWAHLCGCI.

This script wires ``MoWAWindowLatentSampleDataset`` (episode-level latent cache +
window manifest) into a short training loop for the E-004 HLC-GCI module.  It is
intentionally standalone and does not touch ``starVLA/training/train_starvla.py``.

Language is represented with a small learnable embedding table and then projected
to condition tokens.  This is a smoke placeholder: a real run must replace it
with the actual text encoder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from starVLA.dataloader.mowa.episode_latent_store import (
    MoWAEpisodeLatentStoreConfig,
    build_mowa_episode_latent_store,
)
from starVLA.dataloader.mowa.schema import MoWAWindowConfig
from starVLA.dataloader.mowa.window_latent_sample import (
    MoWAWindowLatentSampleDataset,
    WindowLatentSample,
)
from starVLA.dataloader.mowa.window_manifest import (
    MoWAWindowManifestConfig,
    build_mowa_window_manifest,
)
from starVLA.model.modules.mowa.hlcgci import (
    MoWAHLCGCI,
    MoWAHLCGCIConfig,
)


DEFAULT_VIDEO_KEYS = (
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
)


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _episode_indices_from_config(latent_cache: dict[str, Any]) -> tuple[int, ...] | None:
    raw = latent_cache.get("episode_indices")
    if not raw:
        return None
    return tuple(int(value) for value in raw)


def _build_cache_if_needed(
    config: dict[str, Any],
    project_root: Path,
    execute_cache: bool,
) -> Path:
    latent_cache = config.get("latent_cache", {})
    cache_root = _resolve_path(latent_cache["cache_root"], project_root)
    existing = sorted(cache_root.glob("ep_*.h5"))

    if existing and not execute_cache:
        return cache_root

    should_build = execute_cache or not existing
    video_keys = latent_cache.get("video_keys") or list(DEFAULT_VIDEO_KEYS)
    encoder_model_path = latent_cache.get("encoder_model_path")
    store_config = MoWAEpisodeLatentStoreConfig(
        dataset_path=_resolve_path(latent_cache["dataset_path"], project_root),
        cache_root=cache_root,
        video_keys=tuple(video_keys),
        encoder_kind=latent_cache.get("encoder_kind", "fake"),
        encoder_name=latent_cache.get("encoder_name", "mowa-fake-encoder"),
        encoder_version=latent_cache.get("encoder_version", "fake-v1"),
        encoder_model_path=_resolve_path(encoder_model_path, project_root) if encoder_model_path else None,
        latent_model=latent_cache.get("latent_model", "Wan2.2-VAE"),
        latent_model_version=latent_cache.get("latent_model_version", "TBD"),
        latent_type="pooled_vector",
        latent_shape_per_frame=("D",),
        latent_dim=latent_cache.get("latent_dim", 1024),
        # pooled_vector with a Wan VAE returns the raw channel count unless we
        # project.  Mean-pool projection keeps the cache dim aligned with the
        # interface config and avoids hard-coding the VAE output shape.
        flatten_policy=latent_cache.get("flatten_policy", "mean_pool"),
        dtype=latent_cache.get("dtype", "float32"),
        video_backend=latent_cache.get("video_backend", "opencv"),
        vae_batch_size=latent_cache.get("vae_batch_size", 1),
        obs_fps="Data Gate",
        action_hz="Data Gate",
        wam_hz="Data Gate",
        dry_run=not should_build,
        overwrite=execute_cache and bool(latent_cache.get("overwrite", False)),
        episode_indices=_episode_indices_from_config(latent_cache),
    )
    report = build_mowa_episode_latent_store(store_config)
    if report.go_no_go.startswith("No-Go"):
        raise RuntimeError(f"Episode latent store build failed: {report.to_dict()}")
    return cache_root


def _validate_cache_latent_dim(
    cache_root: Path,
    video_keys: tuple[str, ...],
    expected_dim: int,
) -> None:
    store_paths = sorted(cache_root.glob("ep_*.h5"))
    if not store_paths:
        return
    from starVLA.dataloader.mowa.episode_latent_store import MoWAEpisodeLatentStore

    store = MoWAEpisodeLatentStore(store_paths[0])
    for video_key in video_keys:
        if video_key in store.list_video_keys():
            latents = store.get_latents(video_key)
            actual_dim = int(latents.shape[-1])
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Latent dim mismatch for {video_key}: cache has {actual_dim}, "
                    f"interface expects {expected_dim}. Rebuild with "
                    f"--execute-cache or adjust interface dims."
                )
            return
    raise ValueError(f"None of the requested video keys found in cache: {video_keys}")


def _build_manifest(
    config: dict[str, Any],
    cache_root: Path,
) -> Path:
    latent_cache = config.get("latent_cache", {})
    interface = config.get("interface", {})
    manifest_path = cache_root / "window_manifest_e004.parquet"
    window_config = MoWAWindowConfig(
        history_steps=interface.get("history_steps", 10),
        future_steps=latent_cache.get("future_window_steps", 1),
        action_chunk_steps=1,
    )
    video_keys = latent_cache.get("video_keys") or list(DEFAULT_VIDEO_KEYS)
    manifest_config = MoWAWindowManifestConfig(
        cache_root=cache_root,
        output_path=manifest_path,
        window_config=window_config,
        video_keys=tuple(video_keys),
        history_stride=latent_cache.get("history_stride", 1),
        wam_hz=latent_cache.get("wam_hz", 4.0),
        obs_fps="Data Gate",
        action_hz="Data Gate",
        split="train",
        task_name=latent_cache.get("task_name", "TBD"),
        episode_indices=_episode_indices_from_config(latent_cache),
        allow_partial_windows=bool(latent_cache.get("allow_partial_windows", False)),
    )
    report = build_mowa_window_manifest(manifest_config)
    if report.go_no_go.startswith("No-Go"):
        raise RuntimeError(f"Window manifest build failed: {report.to_dict()}")
    return manifest_path


def _collate_e004(batch: list[WindowLatentSample]) -> dict[str, Any]:
    return {
        "history_latents": torch.stack([sample.history_latents for sample in batch]),
        "current_latent": torch.stack([sample.current_latent for sample in batch]),
        "language": [sample.language for sample in batch],
    }


def _build_language_embedding(
    dataset: MoWAWindowLatentSampleDataset,
    condition_hidden_dim: int,
) -> tuple[nn.Embedding, dict[str, int]]:
    languages = sorted({dataset[idx].language for idx in range(len(dataset))})
    if not languages:
        raise ValueError("Dataset contains no language strings.")
    lang_to_idx = {language: idx for idx, language in enumerate(languages)}
    embedding = nn.Embedding(len(languages), condition_hidden_dim)
    return embedding, lang_to_idx


def _build_condition_tokens(
    languages: list[str],
    *,
    embedding: nn.Embedding,
    lang_to_idx: dict[str, int],
    condition_projector: nn.Linear,
    condition_token_count: int,
    condition_hidden_dim: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.tensor(
        [lang_to_idx[language] for language in languages],
        dtype=torch.long,
        device=device,
    )
    language_emb = embedding(indices)  # [B, D]
    projected = condition_projector(language_emb)  # [B, N * D]
    return projected.view(-1, condition_token_count, condition_hidden_dim)


def run_e004_window_latent_train_smoke(
    config: dict[str, Any],
    *,
    execute_cache: bool = False,
    train_steps: int = 4,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run the E-004 HLC-GCI training smoke.

    Parameters
    ----------
    config:
        Loaded YAML config dict.  Must contain ``latent_cache`` and ``interface``
        sections.
    execute_cache:
        If True, (re)materialize the episode latent store.
    train_steps:
        Number of training steps to run.
    batch_size:
        Batch size; defaults to ``interface.batch_size_smoke`` or 2.
    device:
        Torch device to use.
    report_path:
        Optional JSON path to write the smoke report.

    Returns
    -------
    dict
        Smoke report.
    """

    training_started = datetime.now(timezone.utc).isoformat()
    project_root = _project_root()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cache_root = _build_cache_if_needed(config, project_root, execute_cache)
    video_keys = tuple(
        config.get("latent_cache", {}).get("video_keys") or DEFAULT_VIDEO_KEYS
    )
    _validate_cache_latent_dim(
        cache_root,
        video_keys,
        config.get("interface", {}).get("history_latent_dim", 1024),
    )
    manifest_path = _build_manifest(config, cache_root)
    dataset = MoWAWindowLatentSampleDataset(manifest_path)
    if len(dataset) == 0:
        raise ValueError("Window manifest produced no samples.")

    interface = config.get("interface", {})
    model_config = MoWAHLCGCIConfig(
        history_latent_dim=interface.get("history_latent_dim", 1024),
        condition_hidden_dim=interface.get("condition_hidden_dim", 1024),
        history_steps=interface.get("history_steps", 10),
        compressed_history_dim=interface.get("compressed_history_dim", 512),
        gate_hidden_dim=interface.get("gate_hidden_dim", 256),
    )
    condition_token_count = interface.get("condition_token_count", 4)
    model = MoWAHLCGCI(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    language_embedding, lang_to_idx = _build_language_embedding(
        dataset, model_config.condition_hidden_dim
    )
    language_embedding = language_embedding.to(device)
    condition_projector = nn.Linear(
        model_config.condition_hidden_dim,
        condition_token_count * model_config.condition_hidden_dim,
    ).to(device)

    batch_size = batch_size or interface.get("batch_size_smoke", 2)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_e004,
    )

    steps_completed = 0
    loss_value = float("nan")
    final_shapes: dict[str, Any] = {}
    grad_finite = False

    model.train()
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= train_steps:
            break

        history_latent = batch["history_latents"].to(device)
        languages = batch["language"]
        condition_tokens = _build_condition_tokens(
            languages,
            embedding=language_embedding,
            lang_to_idx=lang_to_idx,
            condition_projector=condition_projector,
            condition_token_count=condition_token_count,
            condition_hidden_dim=model_config.condition_hidden_dim,
            device=device,
        )

        output = model(history_latent, condition_tokens)

        # MoWAHLCGCI has no built-in loss.  Use a dummy zero target purely to
        # exercise backprop and shape checks.
        target = torch.zeros_like(output.gated_condition_tokens)
        loss = F.mse_loss(output.gated_condition_tokens, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        steps_completed = batch_index + 1
        grad_finite = all(
            param.grad is None or torch.isfinite(param.grad).all()
            for param in model.parameters()
        )
        final_shapes = {
            "history_latent": list(history_latent.shape),
            "condition_tokens": list(condition_tokens.shape),
            "compressed_history": list(output.compressed_history.shape),
            "gate_values": list(output.gate_values.shape),
            "gated_condition_tokens": list(output.gated_condition_tokens.shape),
        }

        if not torch.isfinite(loss) or not grad_finite:
            break

    loss_is_finite = bool(torch.isfinite(torch.tensor(loss_value)).item())
    go_no_go = (
        "Go"
        if steps_completed == train_steps and loss_is_finite and grad_finite
        else "No-Go"
    )

    report = {
        "experiment_id": config.get("experiment_id", "E-004"),
        "stage": config.get("stage", "hlc_gci"),
        "training_started": training_started,
        "cache_root": str(cache_root),
        "manifest_path": str(manifest_path),
        "device": str(device),
        "steps_completed": steps_completed,
        "train_steps_requested": train_steps,
        "loss_is_finite": loss_is_finite,
        "gradients_finite": grad_finite,
        "final_loss": loss_value,
        "shapes": final_shapes,
        "language_vocab_size": len(lang_to_idx),
        "condition_token_count": condition_token_count,
        "go_no_go": go_no_go,
        "notes": [
            "Language is represented by a learnable embedding table projected to condition tokens; replace with the real text encoder for production.",
            "Loss is a dummy MSE against a zero target because MoWAHLCGCI has no built-in loss.",
        ],
    }

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E-004 HLC-GCI window-latent training smoke."
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=Path("configs/mowa/mowa_e004_wanpi_hlc_gci_candidate.yaml"),
        help="Path to the E-004 WanPI HLC-GCI smoke config.",
    )
    parser.add_argument(
        "--execute-cache",
        action="store_true",
        help="Materialize (or overwrite) the episode latent store before training.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=4,
        help="Number of training steps to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON path to write the smoke report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config_yaml)
    report = run_e004_window_latent_train_smoke(
        config,
        execute_cache=args.execute_cache,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        report_path=args.report_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
