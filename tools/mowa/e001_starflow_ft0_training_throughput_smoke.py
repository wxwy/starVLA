"""Run or validate MoWA E-001 StarFlow save/resume training smoke."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SMOKE_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_training_throughput_smoke.yaml")
SMOKE_CHECK_REPORT = Path("docs_zh/mowa/mowa_e001_starflow_ft0_training_throughput_smoke_check.json")
RUN_ROOT = Path("playground/mowa_ckpt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-001 StarFlow save/resume training smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=SMOKE_CHECK_REPORT)
    parser.add_argument("--execute", action="store_true", help="Actually launch first-run and resume training smoke.")
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_or_validate_starflow_ft0_training_smoke(
        args.repo_root,
        execute=args.execute,
        run_id=args.run_id,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def run_or_validate_starflow_ft0_training_smoke(
    repo_root: Path | str,
    *,
    execute: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    selected_run_id = run_id or _default_run_id()
    commands: list[dict[str, Any]] = []
    if execute:
        commands.append(
            _run_training_command(
                root,
                selected_run_id,
                max_train_steps=1,
                is_resume=False,
            )
        )
        commands.append(
            _run_training_command(
                root,
                selected_run_id,
                max_train_steps=2,
                is_resume=True,
            )
        )
    return build_starflow_ft0_training_throughput_smoke(root, selected_run_id, commands)


def build_starflow_ft0_training_throughput_smoke(
    repo_root: Path | str,
    run_id: str,
    commands: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    run_dir = root / RUN_ROOT / run_id
    checkpoints_dir = run_dir / "checkpoints"
    steps_1 = checkpoints_dir / "steps_1"
    steps_2 = checkpoints_dir / "steps_2"
    final_model = run_dir / "final_model"
    trainer_state_1 = _read_json(steps_1 / "trainer_state.json") or {}
    trainer_state_2 = _read_json(steps_2 / "trainer_state.json") or {}
    config_full = run_dir / "config.full.yaml"
    checks = {
        "smoke_config_created": (root / SMOKE_CONFIG).is_file(),
        "run_root_is_mowa_ckpt": str(run_dir).startswith(str(root / RUN_ROOT)),
        "run_dir_created": run_dir.is_dir(),
        "first_checkpoint_saved": steps_1.is_dir(),
        "resume_checkpoint_saved": steps_2.is_dir(),
        "final_model_saved": final_model.is_dir(),
        "first_trainer_state_step_1": trainer_state_1.get("completed_steps") == 1,
        "resume_trainer_state_step_2": trainer_state_2.get("completed_steps") == 2,
        "optimizer_state_saved_step_1": (steps_1 / "optimizer_rank_00000.pt").is_file(),
        "optimizer_state_saved_step_2": (steps_2 / "optimizer_rank_00000.pt").is_file(),
        "scheduler_state_saved_step_2": (steps_2 / "scheduler.pt").is_file(),
        "rng_state_saved_step_2": any(steps_2.glob("random_states_*.pkl")),
        "config_snapshot_saved": config_full.is_file(),
        "command_runs_succeeded": all(item.get("returncode") == 0 for item in (commands or [])),
    }
    return {
        "stage": "full_heads",
        "experiment_id": "E-001",
        "training_started": bool(commands),
        "save_resume_smoke_executed": bool(commands),
        "checks": checks,
        "configs": {
            "smoke_config": str(SMOKE_CONFIG),
            "run_root": str(RUN_ROOT),
        },
        "observed": {
            "run_id": run_id,
            "run_dir": str(RUN_ROOT / run_id),
            "commands": commands or [],
            "checkpoints": {
                "steps_1": str(RUN_ROOT / run_id / "checkpoints" / "steps_1"),
                "steps_2": str(RUN_ROOT / run_id / "checkpoints" / "steps_2"),
                "final_model": str(RUN_ROOT / run_id / "final_model"),
            },
            "trainer_state_1": trainer_state_1,
            "trainer_state_2": trainer_state_2,
        },
        "unresolved_items": [
            "This is a bounded save/resume smoke, not a full E-001 launch.",
            "Runtime policy is still not confirmed for long training.",
            "The feature source is the gated mowa_future_feature_heads bridge path; "
            "this smoke does not prove action-gain.",
        ],
        "go_no_go": (
            "TBD: save/resume training smoke passed; launch remains gated"
            if all(checks.values())
            else "No-Go: save/resume training smoke incomplete"
        ),
    }


def _run_training_command(root: Path, run_id: str, *, max_train_steps: int, is_resume: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        "starVLA/training/train_starvla.py",
        "--config_yaml",
        str(SMOKE_CONFIG),
        "--run_id",
        run_id,
        "--run_root_dir",
        str(RUN_ROOT),
        "--trainer.max_train_steps",
        str(max_train_steps),
        "--trainer.save_interval",
        "1",
        "--trainer.eval_interval",
        "1000000",
        "--wandb_mode",
        "disabled_for_initial_training",
        "--trainer.is_resume",
        "true" if is_resume else "false",
    ]
    start_time = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed_sec = time.perf_counter() - start_time
    return {
        "phase": "resume" if is_resume else "first_train",
        "returncode": result.returncode,
        "elapsed_sec": elapsed_sec,
        "max_train_steps": max_train_steps,
        "is_resume": is_resume,
        "command": command,
        "tail": result.stdout.splitlines()[-80:],
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _default_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"MoWA-E-001_starflow_ft0_save_resume_smoke_{stamp}"


if __name__ == "__main__":
    main()
