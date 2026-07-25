"""Run bounded full-VLA runtime sweeps for MoWA E-001."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SWEEP_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_training_throughput_smoke.yaml")
SWEEP_REPORT = Path("docs_zh/mowa/mowa_e001_full_vla_runtime_sweep_smoke.json")
RUN_ROOT = Path("playground/mowa_ckpt")
METRICS_ROOT = Path("docs_zh/mowa/runtime_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded MoWA E-001 full-VLA runtime sweep.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=SWEEP_REPORT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-train-steps", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_or_plan_runtime_sweep(
        args.repo_root,
        execute=args.execute,
        batch_sizes=args.batch_sizes,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def run_or_plan_runtime_sweep(
    repo_root: Path | str,
    *,
    execute: bool,
    batch_sizes: list[int],
    max_train_steps: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    root = Path(repo_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs = []
    for batch_size in batch_sizes:
        run_id = f"MoWA-E-001_full_vla_runtime_bs{batch_size}_{stamp}"
        metrics_path = METRICS_ROOT / f"{run_id}.jsonl"
        command = _build_command(
            run_id=run_id,
            batch_size=batch_size,
            max_train_steps=max_train_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            metrics_path=metrics_path,
        )
        result = _run_command(root, command) if execute else None
        metrics = _read_metrics(root / metrics_path)
        runs.append(
            {
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "run_id": run_id,
                "run_dir": str(RUN_ROOT / run_id),
                "runtime_metrics_report": str(metrics_path),
                "command": command,
                "command_result": result,
                "metrics": metrics,
                "summary": summarize_runtime_metrics(metrics),
            }
        )

    checks = {
        "sweep_config_created": (root / SWEEP_CONFIG).is_file(),
        "executed_when_requested": (not execute) or all(
            (run.get("command_result") or {}).get("returncode") == 0 for run in runs
        ),
        "metrics_recorded_when_executed": (not execute) or all(
            bool(run.get("metrics")) for run in runs
        ),
        "peak_vram_recorded_when_executed": (not execute) or all(
            _is_positive_number((run.get("summary") or {}).get("peak_vram_gb"))
            for run in runs
        ),
        "samples_per_sec_recorded_when_executed": (not execute) or all(
            _is_positive_number((run.get("summary") or {}).get("samples_per_sec"))
            for run in runs
        ),
    }
    return {
        "stage": "full_heads",
        "experiment_id": "E-001",
        "training_started": bool(execute),
        "bounded_runtime_sweep": True,
        "full_training_launch": False,
        "checks": checks,
        "configs": {
            "sweep_config": str(SWEEP_CONFIG),
            "run_root": str(RUN_ROOT),
        },
        "runs": runs,
        "unresolved_items": [
            "This is a bounded runtime sweep, not a full E-001 long training launch.",
            "Checkpoint/final-model saving is intentionally skipped for sweep candidates.",
            "Resource policy should only be updated after reviewing stable batch/VRAM/runtime margins.",
        ],
        "go_no_go": (
            "TBD: bounded full-VLA runtime sweep passed; launch remains gated"
            if all(checks.values())
            else "No-Go: bounded full-VLA runtime sweep incomplete"
        ),
    }


def _build_command(
    *,
    run_id: str,
    batch_size: int,
    max_train_steps: int,
    gradient_accumulation_steps: int,
    metrics_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "starVLA/training/train_starvla.py",
        "--config_yaml",
        str(SWEEP_CONFIG),
        "--run_id",
        run_id,
        "--run_root_dir",
        str(RUN_ROOT),
        "--datasets.vla_data.per_device_batch_size",
        str(batch_size),
        "--trainer.max_train_steps",
        str(max_train_steps),
        "--trainer.gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--trainer.save_interval",
        "1000000",
        "--trainer.eval_interval",
        "1000000",
        "--wandb_mode",
        "disabled_for_initial_training",
        "--trainer.is_resume",
        "false",
        "--trainer.skip_final_checkpoint",
        "true",
        "--trainer.runtime_metrics_report",
        str(metrics_path),
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
        "elapsed_sec": time.perf_counter() - start_time,
        "tail": result.stdout.splitlines()[-80:],
    }


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_runtime_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    last = metrics[-1]
    step_times = [row["step_time_sec"] for row in metrics if _is_positive_number(row.get("step_time_sec"))]
    samples = [row["samples_per_sec"] for row in metrics if _is_positive_number(row.get("samples_per_sec"))]
    peak_vram = [row["peak_vram_gb"] for row in metrics if _is_positive_number(row.get("peak_vram_gb"))]
    peak_reserved = [row["peak_reserved_gb"] for row in metrics if _is_positive_number(row.get("peak_reserved_gb"))]
    return {
        "completed_steps": last.get("completed_steps"),
        "total_batch_size": last.get("total_batch_size"),
        "mean_step_time_sec": sum(step_times) / len(step_times) if step_times else None,
        "last_step_time_sec": last.get("step_time_sec"),
        "samples_per_sec": samples[-1] if samples else None,
        "peak_vram_gb": max(peak_vram) if peak_vram else None,
        "peak_reserved_gb": max(peak_reserved) if peak_reserved else None,
        "cuda_device_name": last.get("cuda_device_name"),
    }


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


if __name__ == "__main__":
    main()
