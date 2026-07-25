import json
import os
import logging
import numpy as np
import re
from pathlib import Path
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from torch.utils.data import DataLoader
        import torch.distributed as dist
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        from starVLA.dataloader.mowa.sampler import MoWALatentSamplingConfig, MoWAManifestAnchorSampler
        from starVLA.dataloader.mowa.window_manifest import load_mowa_window_manifest_table
        vla_dataset_cfg = cfg.datasets.vla_data
        if hasattr(cfg, "latent_cache") and cfg.latent_cache:
            latent_cache_cfg = OmegaConf.to_container(cfg.latent_cache, resolve=True)
            existing = vla_dataset_cfg.get("mowa_latent_cache", None)
            # 顶层 latent_cache 也承载 Wan 的 instruction_text_latent；只有
            # 具备视觉 cache 根目录（或数据集已显式声明）时才进入视觉 latent 路径。
            uses_visual_latent_cache = existing is not None or bool(
                latent_cache_cfg.get("cache_root") or latent_cache_cfg.get("manifest_path")
            )
            if uses_visual_latent_cache and existing is not None:
                existing = OmegaConf.to_container(existing, resolve=True)
                latent_cache_cfg = {**latent_cache_cfg, **existing}
            if uses_visual_latent_cache and not latent_cache_cfg.get("manifest_path"):
                cache_root = Path(latent_cache_cfg.get("cache_root", "."))
                manifests = _find_mowa_window_manifests(
                    cache_root,
                    history_window_steps=latent_cache_cfg.get("history_window_steps"),
                    future_window_steps=latent_cache_cfg.get("future_window_steps"),
                )
                if manifests:
                    latent_cache_cfg["manifest_path"] = str(manifests[0])
            if uses_visual_latent_cache:
                vla_dataset_cfg.mowa_latent_cache = latent_cache_cfg

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )
        num_workers = int(getattr(cfg.datasets.vla_data, "num_workers", 0))
        manifest_sampler = None
        latent_sampling = latent_cache_cfg.get("latent_sampling", {}) if "latent_cache_cfg" in locals() else {}
        if latent_sampling.get("sampling_mode") == "task_uniform_episode_uniform_anchor":
            manifest_path = latent_cache_cfg.get("manifest_path")
            if not manifest_path:
                raise ValueError("MoWA latent_sampling requires latent_cache.manifest_path.")
            manifest_sampler = MoWAManifestAnchorSampler(
                vla_dataset._all_steps,
                load_mowa_window_manifest_table(manifest_path),
                MoWALatentSamplingConfig(
                    samples_per_episode_per_epoch=int(latent_sampling.get("samples_per_episode_per_epoch", 1)),
                    regular_anchor_ratio=float(latent_sampling.get("regular_anchor_ratio", 0.80)),
                    terminal_anchor_ratio=float(latent_sampling.get("terminal_anchor_ratio", 0.15)),
                    early_anchor_ratio=float(latent_sampling.get("early_anchor_ratio", 0.05)),
                    replacement=bool(latent_sampling.get("replacement", False)),
                    seed=int(latent_sampling.get("seed", 42)),
                    history_steps=latent_cache_cfg.get("history_window_steps", None),
                    future_steps=latent_cache_cfg.get("future_window_steps", None),
                    action_chunk_steps=latent_cache_cfg.get("action_chunk_steps", None),
                ),
            )
        dataloader_kwargs = {}
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = int(getattr(cfg.datasets.vla_data, "prefetch_factor", 2))
            dataloader_kwargs["pin_memory"] = bool(getattr(cfg.datasets.vla_data, "pin_memory", True))
            dataloader_kwargs["persistent_workers"] = bool(
                getattr(cfg.datasets.vla_data, "persistent_workers", True)
            )
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            sampler=manifest_sampler,
            **dataloader_kwargs,
            # shuffle=True
        )        
        if not dist.is_initialized() or dist.get_rank() == 0:
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        from starVLA.dataloader.vlm_datasets import make_vlm_dataloader
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader


def _find_mowa_window_manifests(
    cache_root: Path,
    history_window_steps=None,
    future_window_steps=None,
) -> list[Path]:
    manifests = _collect_mowa_window_manifests(cache_root)
    if manifests:
        return _rank_mowa_window_manifests(manifests, history_window_steps, future_window_steps)
    sibling_dirs = (
        cache_root.parent / f"{cache_root.name}_manifests",
        cache_root.parent / "window_manifests",
    )
    for sibling_dir in sibling_dirs:
        manifests = sorted(sibling_dir.glob("window_manifest*.parquet"))
        if manifests:
            return _rank_mowa_window_manifests(manifests, history_window_steps, future_window_steps)
    return []


def _collect_mowa_window_manifests(cache_root: Path) -> list[Path]:
    manifests = sorted(cache_root.glob("window_manifest*.parquet"))
    manifests.extend(sorted((cache_root / "window_manifests").glob("*.parquet")))
    return manifests


def _rank_mowa_window_manifests(
    manifests: list[Path],
    history_window_steps=None,
    future_window_steps=None,
) -> list[Path]:
    if history_window_steps is None and future_window_steps is None:
        return manifests
    try:
        history_steps = None if history_window_steps is None else int(history_window_steps)
        future_steps = None if future_window_steps is None else int(future_window_steps)
    except (TypeError, ValueError):
        return manifests
    if history_steps is not None and future_steps is None:
        expected_name = f"window_manifest_h{history_steps}.parquet"
        preferred = [manifest for manifest in manifests if manifest.name == expected_name]
        if preferred:
            return preferred + [manifest for manifest in manifests if manifest not in preferred]

    def _capacity(manifest: Path):
        matched = re.search(r"(?:^|_)h(\d+)_f(\d+)(?:_|\.)", manifest.name)
        if matched is None:
            return None
        return int(matched.group(1)), int(matched.group(2))

    compatible = [
        manifest
        for manifest in manifests
        if (capacity := _capacity(manifest)) is not None
        and (history_steps is None or capacity[0] >= history_steps)
        and (future_steps is None or capacity[1] >= future_steps)
    ]
    if not compatible:
        return manifests
    compatible.sort(key=lambda manifest: (_capacity(manifest), manifest.name))
    return compatible + [manifest for manifest in manifests if manifest not in compatible]
