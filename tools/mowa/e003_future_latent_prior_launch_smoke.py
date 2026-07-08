"""Run the E-003 launch smoke for real Wan2.2 cache plus future latent prior dry-run."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    from tools.mowa.e003_future_latent_prior_train_dry_run import (
        build_e003_future_latent_prior_train_dry_run,
    )
    from tools.mowa.e003_wan2_2_latent_cache_smoke import (
        build_e003_wan2_2_latent_cache_smoke,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from e003_future_latent_prior_train_dry_run import (  # type: ignore[no-redef]
        build_e003_future_latent_prior_train_dry_run,
    )
    from e003_wan2_2_latent_cache_smoke import (  # type: ignore[no-redef]
        build_e003_wan2_2_latent_cache_smoke,
    )


CONFIG = Path("configs/mowa/mowa_e003_future_latent_prior_launch_candidate.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MoWA E-003 launch smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config-yaml", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e003_future_latent_prior_launch_smoke(args.repo_root, args.config_yaml)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e003_future_latent_prior_launch_smoke(
    repo_root: Path | str,
    config_yaml: Path | str = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_path = root / Path(config_yaml)
    cfg = _load_yaml(config_path)
    launch_guard = _select(cfg, "launch_guard") or {}
    latent_cache = _select(cfg, "latent_cache") or {}
    smoke_cfg = _select(cfg, "smoke") or {}
    dry_run_config = Path(smoke_cfg.get("dry_run_config", "configs/mowa/mowa_e003_future_latent_prior_candidate.yaml"))
    with tempfile.TemporaryDirectory(prefix="mowa_e003_launch_smoke_") as temp_dir:
        dry_run_config = _materialize_overwrite_dry_run_config(root, dry_run_config, Path(temp_dir))

        real_wan_smoke = build_e003_wan2_2_latent_cache_smoke(
            root,
            dataset_path=Path(latent_cache.get("dataset_path", "")),
            cache_root=Path(latent_cache.get("cache_root", "")),
            model_path=Path(latent_cache.get("encoder_model_path", "")),
            episode_indices=tuple(latent_cache.get("episode_indices") or ()) or None,
            video_keys=tuple(latent_cache.get("video_keys") or ()) or None,
            execute=True,
        )
        train_dry_run = build_e003_future_latent_prior_train_dry_run(
            root,
            dry_run_config,
            execute_cache=bool(smoke_cfg.get("execute_cache", True)),
        )

    checks = {
        "config_created": config_path.is_file(),
        "launch_guard_open": bool(launch_guard.get("launch_ready")) is True,
        "launch_guard_human_confirmed": bool(launch_guard.get("human_confirmed")) is True,
        "launch_guard_policy_confirmed": bool(launch_guard.get("policy_confirmed")) is True,
        "real_wan_smoke_passed": _report_is_ok(real_wan_smoke),
        "train_dry_run_passed": _report_is_ok(train_dry_run),
    }
    return {
        "stage": "future_latent_prior",
        "task_id": "M3-002",
        "experiment_id": "E-003",
        "training_started": False,
        "launch_ready": bool(launch_guard.get("launch_ready")),
        "checks": checks,
        "config_path": str(Path(config_yaml)),
        "reports": {
            "real_wan_smoke": real_wan_smoke,
            "future_latent_prior_train_dry_run": train_dry_run,
        },
        "observed": {
            "real_wan_cache_root": latent_cache.get("cache_root"),
            "real_wan_encoder_kind": latent_cache.get("encoder_kind"),
            "dry_run_config": smoke_cfg.get("dry_run_config"),
        },
        "unresolved_items": [
            "This launch smoke validates the real Wan2.2 cache path plus the future latent prior dry-run.",
            "The dry-run uses a separate candidate cache path and does not directly consume the real Wan2.2 smoke cache.",
            "It does not replace the formal long-running E-003 training loop yet.",
        ],
        "go_no_go": (
            "TBD: E-003 launch smoke passed; formal long-running training remains gated"
            if all(checks.values())
            else "No-Go: E-003 launch smoke incomplete"
        ),
    }


def _report_is_ok(report: dict[str, Any]) -> bool:
    if not report:
        return False
    if report.get("go_no_go", "").startswith("No-Go"):
        return False
    checks = report.get("checks") or {}
    return bool(checks) and all(bool(value) for value in checks.values())


def _materialize_overwrite_dry_run_config(root: Path, config_path: Path, temp_dir: Path) -> Path:
    source = root / config_path
    cfg = _load_yaml(source)
    if cfg is None:
        return config_path
    latent_cache = _select(cfg, "latent_cache")
    if latent_cache is None:
        return config_path
    OmegaConf.update(cfg, "latent_cache.overwrite", True, merge=True)
    temp_path = temp_dir / source.name
    OmegaConf.save(cfg, temp_path)
    return temp_path


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
