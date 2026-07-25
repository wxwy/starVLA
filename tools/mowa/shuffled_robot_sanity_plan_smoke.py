"""MoWA shuffled-robot sanity plan smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from starVLA.dataloader.mowa import DATA_GATE, build_mowa_shuffled_episode_pairs


CONFIG = Path("configs/mowa/mowa_shuffled_robot_sanity_plan.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_shuffled_robot_sanity_plan_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA shuffled-robot sanity plan smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_shuffled_robot_sanity_plan_smoke(args.repo_root, args.config)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_shuffled_robot_sanity_plan_smoke(
    repo_root: Path | str,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    cfg = OmegaConf.load(root / config_path)
    episode_indices = tuple(int(value) for value in cfg.plan.episode_indices)
    shuffled_pairs = build_mowa_shuffled_episode_pairs(episode_indices)
    checks = {
        "config_exists": (root / config_path).is_file(),
        "pair_count_expected": len(shuffled_pairs) == len(episode_indices),
        "pairs_are_non_self": all(src != dst for src, dst in shuffled_pairs),
        "history_source_robot_only": str(cfg.plan.history_source) == "robot_history_latent_only",
        "shuffle_policy_cyclic_non_self": str(cfg.plan.shuffle_policy) == "cyclic_non_self",
        "future_action_target_only": str(cfg.plan.future_action_policy) == "target_only",
        "rollout_status_data_gate": str(cfg.status.rollout_status) == DATA_GATE,
        "metric_status_data_gate": str(cfg.status.metric_status) == DATA_GATE,
        "leakage_status_data_gate": str(cfg.status.leakage_status) == DATA_GATE,
    }
    return {
        "stage": "hlc_gci",
        "task_id": "M4-003",
        "experiment_id": "E-005",
        "training_started": False,
        "checks": checks,
        "config": str(config_path),
        "observed": {
            "episode_indices": episode_indices,
            "shuffled_pairs": shuffled_pairs,
            "metric_scope": tuple(str(item) for item in cfg.plan.metric_scope),
            "expected_outcome": str(cfg.plan.expected_outcome),
        },
        "unresolved_items": [
            "This smoke validates pair construction only and does not execute rollout.",
            "Real shuffled-robot evidence still depends on trained checkpoints.",
            "Metric drop is an expectation to test later, not a measured result here.",
        ],
        "go_no_go": (
            "TBD: shuffled-robot sanity plan smoke passed; rollout remains gated"
            if all(checks.values())
            else "No-Go: shuffled-robot sanity plan smoke failed"
        ),
    }


if __name__ == "__main__":
    main()
