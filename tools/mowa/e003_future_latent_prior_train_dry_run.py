"""Run a MoWA E-003 future latent prior dry-run on fake cache batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from starVLA.dataloader.mowa import (
    DATA_GATE,
    MoWAFutureLatentCacheBuildConfig,
    MoWAFutureLatentCacheDataset,
    build_mowa_future_latent_cache,
    validate_mowa_future_latent_cache,
)
from starVLA.model.modules.mowa import (
    MoWAFutureLatentPrior,
    MoWAFutureLatentPriorConfig,
)


CONFIG = Path("configs/mowa/smoke/mowa_e003_future_latent_prior_candidate.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-003 dry-run on fake cache batches.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config-yaml", type=Path, default=CONFIG)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--execute-cache",
        action="store_true",
        help="Materialize the fake latent cache before the model dry-run.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e003_future_latent_prior_train_dry_run(
        args.repo_root,
        args.config_yaml,
        dataset_path=args.dataset_path,
        cache_root=args.cache_root,
        execute_cache=args.execute_cache,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e003_future_latent_prior_train_dry_run(
    repo_root: Path | str,
    config_yaml: Path | str = CONFIG,
    *,
    dataset_path: Path | str | None = None,
    cache_root: Path | str | None = None,
    execute_cache: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_path = root / Path(config_yaml)
    cfg = _load_yaml(config_path)
    latent_cache = _select(cfg, "latent_cache") or {}
    interface = _select(cfg, "interface") or {}
    status = _select(cfg, "status") or {}

    resolved_dataset_path = Path(dataset_path) if dataset_path is not None else Path(latent_cache.get("dataset_path", ""))
    resolved_cache_root = Path(cache_root) if cache_root is not None else Path(latent_cache.get("cache_root", ""))
    build_config = MoWAFutureLatentCacheBuildConfig(
        dataset_path=resolved_dataset_path,
        cache_root=resolved_cache_root,
        encoder_name=str(latent_cache.get("encoder_name", "wan-fake-encoder")),
        encoder_version=str(latent_cache.get("encoder_version", "fake-v1")),
        encoder_kind=str(latent_cache.get("encoder_kind", "fake")),
        encoder_model_path=(
            Path(latent_cache["encoder_model_path"])
            if latent_cache.get("encoder_model_path")
            else None
        ),
        latent_dim=int(latent_cache.get("latent_dim", interface.get("future_latent_dim", 1024))),
        video_keys=tuple(latent_cache.get("video_keys") or ()),
        current_window_steps=int(latent_cache.get("current_window_steps", 1)),
        future_window_steps=int(latent_cache.get("future_window_steps", 8)),
        history_window_steps=int(latent_cache.get("history_window_steps", 8)),
        episode_indices=tuple(latent_cache.get("episode_indices") or ()) or None,
        anchor_mode=str(latent_cache.get("anchor_mode", "smoke")),
        allow_partial_windows=bool(latent_cache.get("allow_partial_windows", False)),
        overwrite=bool(latent_cache.get("overwrite", False)),
        dry_run=not execute_cache,
    )

    cache_build_report = build_mowa_future_latent_cache(build_config).to_dict()
    cache_validation_report = (
        validate_mowa_future_latent_cache(resolved_cache_root).to_dict()
        if execute_cache
        else None
    )

    dataset = MoWAFutureLatentCacheDataset(resolved_cache_root) if execute_cache else None
    sample = dataset[0] if dataset is not None and len(dataset) else None

    model = MoWAFutureLatentPrior(
        MoWAFutureLatentPriorConfig(
            current_latent_dim=int(interface.get("current_latent_dim", 1024)),
            text_hidden_dim=int(interface.get("text_hidden_dim", 2048)),
            hidden_dim=int(interface.get("hidden_dim", 2048)),
            future_latent_dim=int(interface.get("future_latent_dim", 1024)),
        )
    )
    batch_size = int(interface.get("batch_size_smoke", 2))
    generator = torch.Generator().manual_seed(0)
    if sample is not None:
        current_latent = sample["current_latent"].detach().clone().unsqueeze(0).repeat(batch_size, 1)
        future_latent_target = sample["future_latent"].detach().clone().unsqueeze(0).repeat(batch_size, 1)
        batch_source = "fake_cache_sample"
    else:
        current_latent = torch.randn(
            batch_size,
            int(interface.get("current_latent_dim", 1024)),
            generator=generator,
        )
        future_latent_target = torch.randn(
            batch_size,
            int(interface.get("future_latent_dim", 1024)),
            generator=generator,
        )
        batch_source = "random_smoke"
    text_hidden = torch.randn(
        batch_size,
        int(interface.get("text_hidden_dim", 2048)),
        generator=generator,
    )

    loss, losses, output = model.compute_loss(current_latent, text_hidden, future_latent_target)
    history_rejected = False
    history_reject_message = None
    try:
        model.forward(
            current_latent,
            text_hidden,
            history_latent=torch.randn(
                batch_size,
                int(interface.get("future_latent_dim", 1024)),
                generator=generator,
            ),
        )
    except ValueError as exc:
        history_rejected = True
        history_reject_message = str(exc)

    checks = {
        "config_created": config_path.is_file(),
        "launch_guard_closed": _select(cfg, "launch_guard.launch_ready") is False,
        "cache_build_report_ok": cache_build_report.get("written_artifact_count", 0)
        > 0 or cache_build_report.get("planned_artifact_count", 0) > 0,
        "cache_validation_ok": (
            True if cache_validation_report is None else bool(cache_validation_report.get("all_ok"))
        ),
        "cache_dataset_sample_available": sample is not None or not execute_cache,
        "model_forward_succeeds": tuple(output.predicted_future_latent.shape)
        == (batch_size, int(interface.get("future_latent_dim", 1024))),
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "history_latent_rejected": history_rejected,
        "history_latent_status_data_gate": status.get("history_latent_status", DATA_GATE) == DATA_GATE,
        "latent_shape_status_data_gate": status.get("latent_shape_status", DATA_GATE) == DATA_GATE,
    }
    return {
        "stage": "future_latent_prior",
        "task_id": "M3-002",
        "experiment_id": "E-003",
        "training_started": False,
        "execute_cache": execute_cache,
        "batch_source": batch_source,
        "checks": checks,
        "reports": {
            "config": str(Path(config_yaml)),
            "cache_build_report": cache_build_report,
            "cache_validation_report": cache_validation_report,
        },
        "observed": {
            "current_latent_shape": tuple(current_latent.shape),
            "text_hidden_shape": tuple(text_hidden.shape),
            "future_latent_target_shape": tuple(future_latent_target.shape),
            "predicted_future_latent_shape": tuple(output.predicted_future_latent.shape),
            "loss": float(loss.item()),
            "loss_keys": tuple(losses.keys()),
            "history_reject_message": history_reject_message,
            "cache_sample_count": len(dataset) if dataset is not None else 0,
        },
        "unresolved_items": [
            "This dry-run uses the fake latent cache builder, not the real Wan encoder/VAE stack.",
            "No formal training loop, checkpoint save, or resume logic is executed here.",
            "Future latent remains target-only and is not valid as WAM input evidence.",
        ],
        "go_no_go": (
            "TBD: E-003 cache dry-run passed; real Wan cache builder remains gated"
            if all(checks.values())
            else "No-Go: E-003 cache dry-run incomplete"
        ),
    }


def _load_yaml(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return OmegaConf.load(path)


def _select(cfg: Any | None, dot_path: str) -> Any:
    if cfg is None:
        return None
    value = OmegaConf.select(cfg, dot_path, default=None)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


if __name__ == "__main__":
    main()
