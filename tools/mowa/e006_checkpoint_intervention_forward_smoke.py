"""Run MoWA E-006 checkpoint-backed intervention forward smoke."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tools.mowa.mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference
except ModuleNotFoundError:
    from mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference


CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_full_path_dry_run.yaml")
E006_EVAL_LOAD_CONFIG = Path("configs/mowa/mowa_e006_eval_load_smoke.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_e006_checkpoint_intervention_forward_smoke.json")
PER_INTERVENTION_DIR = Path("docs_zh/mowa/e006_checkpoint_intervention_forward")
RUN_ROOT = Path("playground/mowa_ckpt")
INTERVENTIONS = (
    "baseline",
    "zero",
    "batch_shuffle",
    "head_mask_control",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-006 checkpoint intervention forward smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_or_plan_checkpoint_intervention_forward_smoke(
        args.repo_root,
        checkpoint=args.checkpoint,
        execute=args.execute,
        batch_size=args.batch_size,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def run_or_plan_checkpoint_intervention_forward_smoke(
    repo_root: Path | str,
    *,
    checkpoint: Path | None,
    execute: bool,
    batch_size: int,
) -> dict[str, Any]:
    root = Path(repo_root)
    checkpoint = checkpoint or _load_default_checkpoint_path(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs = []
    for intervention in INTERVENTIONS:
        report_path = PER_INTERVENTION_DIR / f"{stamp}_{intervention}.json"
        run_id = f"MoWA-E-006_checkpoint_forward_{intervention}_{stamp}"
        command = build_command(
            checkpoint=checkpoint,
            intervention=intervention,
            run_id=run_id,
            report_path=report_path,
            batch_size=batch_size,
        )
        command_result = _run_command(root, command) if execute else None
        report = _read_json(root / report_path) or {}
        runs.append(
            {
                "intervention": intervention,
                "run_id": run_id,
                "run_dir": str(RUN_ROOT / run_id),
                "report_path": str(report_path),
                "command": command,
                "command_result": command_result,
                "forward": report.get("forward", {}),
                "checkpoint_load": report.get("checkpoint_load", {}),
            }
        )

    checks = _build_checks(root, checkpoint, execute, runs, batch_size)
    return {
        "stage": "M5",
        "task_id": "M5-006",
        "experiment_id": "E-006",
        "experiment_name": "checkpoint-backed WAM-to-action intervention forward smoke",
        "training_started": False,
        "eval_started": False,
        "execute_requested": bool(execute),
        "checkpoint": str(checkpoint),
        "batch_size": int(batch_size),
        "interventions": list(INTERVENTIONS),
        "checks": checks,
        "runs": runs,
        "action_loss_delta_vs_baseline": _action_loss_deltas(runs),
        "unresolved_items": [
            "This is one real RoboCasa batch forward smoke with checkpoint weights, not environment rollout.",
            "Action success deltas still require policy rollout.",
            "Long E-001 training remains gated by launch/runtime policy confirmation.",
        ],
        "go_no_go": (
            "TBD: checkpoint-backed intervention forward smoke passed; policy rollout remains pending"
            if all(checks.values())
            else "No-Go: checkpoint-backed intervention forward smoke incomplete"
        ),
    }


def build_command(
    *,
    checkpoint: Path,
    intervention: str,
    run_id: str,
    report_path: Path,
    batch_size: int,
) -> list[str]:
    return [
        sys.executable,
        "starVLA/training/train_starvla.py",
        "--config_yaml",
        str(CONFIG),
        "--run_id",
        run_id,
        "--run_root_dir",
        str(RUN_ROOT),
        "--datasets.vla_data.per_device_batch_size",
        str(batch_size),
        "--framework.mowa.layerwise_bridge_token_intervention",
        intervention,
        "--trainer.full_path_dry_run_load_checkpoint",
        "true",
        "--trainer.full_path_dry_run_checkpoint",
        str(checkpoint),
        "--trainer.full_path_dry_run_report",
        str(report_path),
        "--wandb_mode",
        "disabled_for_initial_training",
    ]


def _run_command(root: Path, command: list[str]) -> dict[str, Any]:
    start_time = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "elapsed_sec": float(time.perf_counter() - start_time),
        "tail": result.stdout.splitlines()[-80:],
    }


def _build_checks(
    root: Path,
    checkpoint: Path,
    execute: bool,
    runs: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, bool]:
    return {
        "config_exists": (root / CONFIG).is_file(),
        "checkpoint_exists": (root / checkpoint).is_dir(),
        "checkpoint_under_mowa_ckpt": str(checkpoint).startswith("playground/mowa_ckpt/"),
        "batch_size_allows_shuffle": batch_size > 1,
        "executed_when_requested": (not execute) or all(
            (run.get("command_result") or {}).get("returncode") == 0 for run in runs
        ),
        "reports_written_when_executed": (not execute) or all(
            (root / run["report_path"]).is_file() for run in runs
        ),
        "checkpoint_loaded_when_executed": (not execute) or all(
            (run.get("checkpoint_load") or {}).get("loaded") is True for run in runs
        ),
        "forward_evaluated_when_executed": (not execute) or all(
            (run.get("forward") or {}).get("evaluated") is True for run in runs
        ),
        "intervention_metadata_recorded_when_executed": (not execute) or all(
            (run.get("forward") or {}).get("mowa_layerwise_bridge_intervention") == run["intervention"]
            for run in runs
        ),
        "action_loss_recorded_when_executed": (not execute) or all(
            _is_number((run.get("forward") or {}).get("action_loss")) for run in runs
        ),
        "batch_shuffle_applied_when_executed": (not execute) or (
            (next(run for run in runs if run["intervention"] == "batch_shuffle").get("forward") or {}).get(
                "mowa_layerwise_bridge_intervention_applied"
            )
            is True
        ),
    }


def _action_loss_deltas(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    baseline = _action_loss(next((run for run in runs if run["intervention"] == "baseline"), {}))
    deltas = {}
    for run in runs:
        loss = _action_loss(run)
        deltas[run["intervention"]] = None if baseline is None or loss is None else float(loss - baseline)
    return deltas


def _action_loss(run: dict[str, Any]) -> float | None:
    value = (run.get("forward") or {}).get("action_loss")
    return float(value) if _is_number(value) else None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_default_checkpoint_path(repo_root: Path) -> Path:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(repo_root / E006_EVAL_LOAD_CONFIG)
    checkpoint_root_policy = Path(
        str(getattr(cfg.checkpoint, "checkpoint_root_policy", "playground/mowa_ckpt"))
    )
    return resolve_mowa_checkpoint_reference(
        repo_root,
        cfg.checkpoint.eval_candidate_checkpoint,
        checkpoint_root_policy=checkpoint_root_policy,
    )


if __name__ == "__main__":
    main()
