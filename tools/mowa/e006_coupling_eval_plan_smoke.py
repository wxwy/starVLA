"""MoWA E-006 coupling eval plan smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


E006_PLAN_CONFIG = Path("configs/mowa/mowa_e006_coupling_eval_plan.yaml")
ACTION_BRIDGE_CONFIG = Path("configs/mowa/mowa_action_bridge_interface.yaml")
READINESS_REPORT = Path("docs_zh/mowa/mowa_e001_readiness_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-006 coupling eval plan smoke.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. When omitted, prints to stdout.",
    )
    return parser.parse_args()


def build_e006_coupling_eval_plan_smoke(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    readiness = _read_json(root / READINESS_REPORT) or {}
    readiness_checks = readiness.get("checks", {}) if isinstance(readiness, dict) else {}

    checks = {
        "plan_config_created": (root / E006_PLAN_CONFIG).is_file(),
        "action_bridge_interface_created": (root / ACTION_BRIDGE_CONFIG).is_file(),
        "readiness_report_created": (root / READINESS_REPORT).is_file(),
        "readiness_bridge_check_passed": readiness_checks.get("action_bridge_interface_created") is True,
        "training_not_started": readiness.get("training_started") is False,
    }
    interventions = [
        "baseline_bridge_tokens_enabled",
        "feature_removal_bridge_tokens_zeroed",
        "batch_shuffle_bridge_tokens_shuffled_across_batch",
        "head_mask_control_constructible_heads_only",
    ]
    metrics = {
        "action_success_delta": "TBD",
        "action_loss_delta": "TBD",
        "bridge_token_norm": "TBD",
        "latency_cost": "TBD",
        "gpu_memory_delta": "TBD",
    }
    unresolved_items = [
        "E-006 eval requires a trained or smoke-compatible action checkpoint",
        "E-001 launch draft remains not executable",
        "production WAM Hz/window remain Data Gate",
        "runtime policy draft not confirmed",
    ]
    ready_for_eval_plan = all(checks.values())
    return {
        "stage": "M5",
        "task_id": "M5-002",
        "experiment_id": "E-006",
        "experiment_name": "WAM-to-action coupling",
        "training_started": False,
        "eval_started": False,
        "checks": checks,
        "interventions": interventions,
        "metrics": metrics,
        "guardrails": {
            "modify_layerwisefm_internal_logic": False,
            "future_action_as_wam_input": False,
            "planner_fsm_rl_scene_graph_baselines": False,
            "per_head_leave_one_out_sweep": False,
            "claim_action_gain_without_feature_removal": False,
        },
        "unresolved_items": unresolved_items,
        "go_no_go": (
            "TBD: E-006 eval plan available; checkpoint/runtime remain Data Gate"
            if ready_for_eval_plan
            else "No-Go: E-006 eval plan prerequisites incomplete"
        ),
        "notes": [
            "This smoke only validates the eval plan and does not run E-006.",
            "Feature removal or shuffle evidence is required before claiming action gains from MoWA bridge features.",
        ],
    }


def main() -> None:
    args = parse_args()
    payload = build_e006_coupling_eval_plan_smoke(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
