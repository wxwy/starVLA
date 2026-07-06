"""Offline diagnostic for the E-001 steps_1000 checkpoint pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from tools.mowa.e006_checkpoint_intervention_forward_smoke import (
    run_or_plan_checkpoint_intervention_forward_smoke,
)
from tools.mowa.e006_eval_load_smoke import build_e006_eval_load_smoke


OUTPUT = Path("docs_zh/mowa/mowa_e001_steps_1000_offline_diagnostic.json")
ROLLOUT_REPORT = Path("docs_zh/mowa/mowa_e006_policy_rollout_smoke.json")
BASELINE_ROLLOUT_REPORT = Path("docs_zh/mowa/mowa_e001_baseline_policy_rollout_smoke.json")
FORWARD_REPORT = Path("docs_zh/mowa/mowa_e006_checkpoint_intervention_forward_smoke.json")
DEFAULT_CHECKPOINT = Path(
    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_bs4_candidate/checkpoints/steps_1000"
)
DEFAULT_BASELINE_CHECKPOINT = Path(
    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_baseline_bs4_candidate/checkpoints/steps_1000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 steps_1000 offline diagnostic.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--execute-forward-smoke", action="store_true")
    parser.add_argument("--execute-eval-load", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e001_steps_1000_offline_diagnostic(
        args.repo_root,
        checkpoint=args.checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        batch_size=args.batch_size,
        execute_forward_smoke=args.execute_forward_smoke,
        execute_eval_load=args.execute_eval_load,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


def build_e001_steps_1000_offline_diagnostic(
    repo_root: Path | str,
    *,
    checkpoint: Path,
    baseline_checkpoint: Path,
    batch_size: int,
    execute_forward_smoke: bool,
    execute_eval_load: bool,
) -> dict[str, Any]:
    root = Path(repo_root)
    checkpoint_ref = _make_repo_relative(root, checkpoint)
    baseline_checkpoint_ref = _make_repo_relative(root, baseline_checkpoint)
    eval_load = build_e006_eval_load_smoke(
        root,
        checkpoint=checkpoint_ref,
        final_model=checkpoint_ref.parents[1] / "final_model",
        execute_load=execute_eval_load,
    )
    baseline_eval_load = build_e006_eval_load_smoke(
        root,
        checkpoint=baseline_checkpoint_ref,
        final_model=baseline_checkpoint_ref.parents[1] / "final_model",
        execute_load=execute_eval_load,
    )
    forward = _load_existing_forward_report(root, checkpoint_ref)
    if forward is None or execute_forward_smoke:
        forward = run_or_plan_checkpoint_intervention_forward_smoke(
            root,
            checkpoint=checkpoint_ref,
            execute=execute_forward_smoke,
            batch_size=batch_size,
        )
    rollout = _read_json(root / ROLLOUT_REPORT) or {}
    baseline_rollout = _read_json(root / BASELINE_ROLLOUT_REPORT) or {}
    diagnosis = _diagnose_checkpoint_behavior(
        forward,
        rollout,
        baseline_rollout=baseline_rollout,
    )
    checks = {
        "checkpoint_exists": (root / checkpoint_ref).is_dir(),
        "baseline_checkpoint_exists": (root / baseline_checkpoint_ref).is_dir(),
        "eval_load_ready": bool(eval_load.get("checks", {}).get("checkpoint_dir_exists")),
        "baseline_eval_load_ready": bool(
            baseline_eval_load.get("checks", {}).get("checkpoint_dir_exists")
        ),
        "forward_smoke_ready": bool(forward.get("checks", {}).get("checkpoint_exists")),
        "forward_interventions_cover_four_modes": set(forward.get("interventions") or []) == {
            "baseline",
            "zero",
            "batch_shuffle",
            "head_mask_control",
        },
        "rollout_report_present": (root / ROLLOUT_REPORT).is_file(),
        "baseline_rollout_report_present": (root / BASELINE_ROLLOUT_REPORT).is_file(),
        "rollout_was_executed": rollout.get("eval_started") is True,
        "baseline_rollout_was_executed": baseline_rollout.get("eval_started") is True,
    }
    return {
        "stage": "P0",
        "experiment_id": "E-001",
        "diagnostic_name": "steps_1000_offline_diagnostic",
        "training_started": False,
        "eval_started": False,
        "checkpoint": str(checkpoint_ref),
        "baseline_checkpoint": str(baseline_checkpoint_ref),
        "batch_size": int(batch_size),
        "execute_forward_smoke": bool(execute_forward_smoke),
        "execute_eval_load": bool(execute_eval_load),
        "checks": checks,
        "eval_load": {
            "go_no_go": eval_load.get("go_no_go"),
            "checks": eval_load.get("checks"),
            "observed": eval_load.get("observed"),
        },
        "baseline_eval_load": {
            "go_no_go": baseline_eval_load.get("go_no_go"),
            "checks": baseline_eval_load.get("checks"),
            "observed": baseline_eval_load.get("observed"),
        },
        "forward": {
            "go_no_go": forward.get("go_no_go"),
            "checks": forward.get("checks"),
            "action_loss_delta_vs_baseline": forward.get("action_loss_delta_vs_baseline"),
            "runs": [
                {
                    "intervention": run.get("intervention"),
                    "action_loss": (run.get("forward") or {}).get("action_loss"),
                    "bridge_active_heads": (run.get("forward") or {}).get(
                        "mowa_layerwise_bridge_active_heads"
                    ),
                    "bridge_feature_source": (run.get("forward") or {}).get(
                        "mowa_layerwise_bridge_feature_source"
                    ),
                    "intervention_applied": (run.get("forward") or {}).get(
                        "mowa_layerwise_bridge_intervention_applied"
                    ),
                    "intervention_note": (run.get("forward") or {}).get(
                        "mowa_layerwise_bridge_intervention_note"
                    ),
                }
                for run in forward.get("runs") or []
            ],
        },
        "rollout": {
            "go_no_go": rollout.get("go_no_go"),
            "checks": rollout.get("checks"),
            "success_rates": _extract_rollout_success_rates(rollout),
        },
        "baseline_rollout": {
            "go_no_go": baseline_rollout.get("go_no_go"),
            "checks": baseline_rollout.get("checks"),
            "success_rates": _extract_rollout_success_rates(baseline_rollout),
        },
        "diagnosis": diagnosis,
        "go_no_go": (
            "TBD: offline diagnostic completed; checkpoint remains usable for analysis"
            if all(checks.values())
            else "No-Go: offline diagnostic prerequisites incomplete"
        ),
    }


def _make_repo_relative(root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else (root / path)
    resolved = resolved.resolve()
    return resolved.relative_to(root.resolve())


def _extract_rollout_success_rates(report: dict[str, Any]) -> dict[str, float | None]:
    success_rates: dict[str, float | None] = {}
    for run in report.get("runs") or []:
        intervention = run.get("intervention")
        rollout_result = run.get("rollout_result") or {}
        success_rate = rollout_result.get("success_rate")
        success_rates[str(intervention)] = (
            float(success_rate) if isinstance(success_rate, (int, float)) else None
        )
    return success_rates


def _load_existing_forward_report(root: Path, checkpoint: Path) -> dict[str, Any] | None:
    report = _read_json(root / FORWARD_REPORT)
    if report is None:
        return None
    return report if report.get("checkpoint") == str(checkpoint) else None


def _diagnose_checkpoint_behavior(
    forward_report: dict[str, Any],
    rollout_report: dict[str, Any],
    *,
    baseline_rollout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deltas = forward_report.get("action_loss_delta_vs_baseline") or {}
    non_baseline_deltas = {
        key: float(value)
        for key, value in deltas.items()
        if key != "baseline" and isinstance(value, (int, float))
    }
    max_abs_delta = max((abs(value) for value in non_baseline_deltas.values()), default=None)
    success_rates = _extract_rollout_success_rates(rollout_report)
    max_success_rate = max(
        (value for value in success_rates.values() if isinstance(value, (int, float))),
        default=None,
    )
    baseline_success_rates = _extract_rollout_success_rates(baseline_rollout or {})
    baseline_max_success_rate = max(
        (value for value in baseline_success_rates.values() if isinstance(value, (int, float))),
        default=None,
    )
    forward_sensitivity = _classify_forward_sensitivity(max_abs_delta)
    verdict = _classify_diagnostic_verdict(
        forward_sensitivity=forward_sensitivity,
        max_success_rate=max_success_rate,
        baseline_max_success_rate=baseline_max_success_rate,
    )
    return {
        "forward_sensitivity": forward_sensitivity,
        "max_abs_action_loss_delta_vs_baseline": max_abs_delta,
        "max_rollout_success_rate": max_success_rate,
        "baseline_max_rollout_success_rate": baseline_max_success_rate,
        "mowa_minus_baseline_max_success_rate": (
            None
            if max_success_rate is None or baseline_max_success_rate is None
            else float(max_success_rate - baseline_max_success_rate)
        ),
        "verdict": verdict,
        "likely_causes": _likely_causes(
            forward_sensitivity=forward_sensitivity,
            max_success_rate=max_success_rate,
            baseline_max_success_rate=baseline_max_success_rate,
        ),
    }


def _classify_forward_sensitivity(max_abs_delta: float | None) -> str:
    if max_abs_delta is None:
        return "not_available"
    if max_abs_delta < 1e-5:
        return "no_observable_effect"
    if max_abs_delta < 1e-3:
        return "very_weak_effect"
    return "observable_effect"


def _classify_diagnostic_verdict(
    *,
    forward_sensitivity: str,
    max_success_rate: float | None,
    baseline_max_success_rate: float | None,
) -> str:
    if baseline_max_success_rate is not None and max_success_rate is not None:
        if baseline_max_success_rate == 0.0 and max_success_rate == 0.0:
            return "baseline_and_mowa_both_zero_success"
        if max_success_rate > baseline_max_success_rate:
            return "mowa_exceeds_baseline_rollout"
        if max_success_rate < baseline_max_success_rate:
            return "mowa_underperforms_baseline_rollout"
        if max_success_rate == baseline_max_success_rate:
            return "mowa_matches_baseline_rollout"
    if max_success_rate is None:
        return "forward_only_diagnostic_available"
    if max_success_rate == 0.0 and forward_sensitivity == "observable_effect":
        return "bridge_is_wired_but_checkpoint_has_no_action_gain"
    if max_success_rate == 0.0 and forward_sensitivity in {"very_weak_effect", "no_observable_effect"}:
        return "checkpoint_is_undertrained_and_bridge_effect_is_weak"
    if max_success_rate > 0.0:
        return "checkpoint_shows_nonzero_rollout_success"
    return "diagnostic_inconclusive"


def _likely_causes(
    *,
    forward_sensitivity: str,
    max_success_rate: float | None,
    baseline_max_success_rate: float | None,
) -> list[str]:
    if baseline_max_success_rate == 0.0 and max_success_rate == 0.0:
        return [
            "Baseline 和 MoWA rollout 都是 0，当前不能把问题归因到 bridge 本身。",
            (
                "先检查训练步数、OpenDrawer 数据域难度、"
                "监督强度和动作分布，而不是继续扩展 E-002/P1。"
            ),
            (
                "MoWA forward intervention 仍可用于确认桥接是否接通，"
                "但它不能替代 baseline 对照下的策略有效性证据。"
            ),
        ]
    if max_success_rate == 0.0 and forward_sensitivity == "observable_effect":
        return [
            "MoWA bridge interventions change action loss on a real batch, so the bridge path is wired.",
            (
                "The steps_1000 checkpoint still fails rollout, "
                "which points to undertraining or weak policy quality rather than a dead bridge."
            ),
            (
                "Next diagnostics should focus on longer training, small-set overfit, "
                "or stronger supervision scaling instead of bridge plumbing."
            ),
        ]
    if max_success_rate == 0.0 and forward_sensitivity in {"very_weak_effect", "no_observable_effect"}:
        return [
            "Bridge interventions barely change action loss on a real batch.",
            (
                "The checkpoint is likely too weak for action gain, "
                "and the current bridge signal may still be drowned out."
            ),
            (
                "Check supervision scale, bridge token magnitude, "
                "and whether a longer run increases intervention sensitivity."
            ),
        ]
    if max_success_rate is None:
        return [
            "Forward diagnostic is available, but rollout evidence is missing.",
        ]
    return [
        "Checkpoint behavior does not match the common failure templates; inspect per-intervention logs.",
    ]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
