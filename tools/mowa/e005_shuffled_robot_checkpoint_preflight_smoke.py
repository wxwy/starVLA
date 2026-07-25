"""Validate the E-005 shuffled-robot checkpoint-backed preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from starVLA.dataloader.mowa import build_mowa_shuffled_episode_pairs

try:
    from tools.mowa.mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference


CONFIG = Path("configs/mowa/mowa_e005_shuffled_robot_checkpoint_candidate.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_e005_shuffled_robot_checkpoint_preflight_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-005 checkpoint preflight smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e005_shuffled_robot_checkpoint_preflight_smoke(args.repo_root, args.config)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e005_shuffled_robot_checkpoint_preflight_smoke(
    repo_root: Path | str,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_file = root / config_path
    cfg = OmegaConf.load(config_file)
    plan_cfg = _select(cfg, "references.plan_config")
    plan_smoke = _select(cfg, "references.plan_smoke")
    plan = OmegaConf.load(root / Path(plan_cfg)) if plan_cfg else None
    plan_smoke_report = _read_json(root / Path(plan_smoke)) if plan_smoke else None
    episode_indices = tuple(int(value) for value in (plan.plan.episode_indices if plan is not None else ()))
    shuffled_pairs = build_mowa_shuffled_episode_pairs(episode_indices)
    checkpoint = resolve_mowa_checkpoint_reference(
        root,
        _select(cfg, "checkpoint.checkpoint"),
        checkpoint_root_policy=_select(cfg, "checkpoint.checkpoint_root_policy"),
    )
    checkpoint_path = root / checkpoint
    checks = {
        "config_exists": config_file.is_file(),
        "plan_config_exists": plan is not None,
        "plan_smoke_report_exists": plan_smoke_report is not None,
        "pair_count_expected": len(shuffled_pairs) == len(episode_indices),
        "pairs_are_non_self": all(src != dst for src, dst in shuffled_pairs),
        "checkpoint_exists": checkpoint_path.is_dir(),
        "checkpoint_under_mowa_ckpt": str(checkpoint).startswith(
            str(_select(cfg, "checkpoint.checkpoint_root_policy"))
        ),
        "plan_shuffle_policy_cyclic_non_self": _select(cfg, "rollout.shuffle_policy") == "cyclic_non_self",
        "plan_future_action_target_only": _select(cfg, "rollout.future_action_policy") == "target_only",
        "n_envs_gt_1": int(_select(cfg, "rollout.n_envs") or 0) > 1,
        "n_episodes_covers_vector_envs": int(_select(cfg, "rollout.n_episodes") or 0)
        >= int(_select(cfg, "rollout.n_envs") or 0),
    }
    checks["checkpoint_preflight_passed"] = all(checks.values())
    return {
        "stage": "hlc_gci",
        "task_id": "M4-003",
        "experiment_id": "E-005",
        "training_started": False,
        "launch_ready": False,
        "checks": checks,
        "config_path": str(config_path),
        "observed": {
            "checkpoint": str(checkpoint),
            "episode_indices": episode_indices,
            "shuffled_pairs": shuffled_pairs,
        },
        "unresolved_items": [
            "This preflight validates shuffled-robot checkpoint wiring only; it does not launch rollout.",
            "Real shuffled-robot evidence still depends on a meaningful E-004 checkpoint and downstream rollout report.",
        ],
        "go_no_go": (
            "TBD: E-005 shuffled-robot checkpoint preflight passed; rollout remains gated"
            if all(checks.values())
            else "No-Go: E-005 shuffled-robot checkpoint preflight incomplete"
        ),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _select(cfg: Any | None, dot_path: str) -> Any:
    if cfg is None:
        return None
    value = OmegaConf.select(cfg, dot_path, default=None)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


if __name__ == "__main__":
    main()
