"""Run the E-005 shuffled-robot checkpoint-backed policy rollout smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    from tools.mowa.mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference
    from tools.mowa.e006_policy_rollout_smoke import (
        _build_rollout_commands,
        _collect_rollout_result,
        _command_port,
        _extract_rollout_blocker,
        _is_number,
        _run_client,
        _server_failure_category,
        _wait_for_port,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference
    from e006_policy_rollout_smoke import (  # type: ignore[no-redef]
        _build_rollout_commands,
        _collect_rollout_result,
        _command_port,
        _extract_rollout_blocker,
        _is_number,
        _run_client,
        _server_failure_category,
        _wait_for_port,
    )

import subprocess
import time
from datetime import datetime


ROLLOUT_CONFIG = Path("configs/mowa/mowa_e005_shuffled_robot_rollout_candidate.yaml")
ROLLOUT_OUTPUT = Path("docs_zh/mowa/mowa_e005_shuffled_robot_rollout_smoke.json")
ARTIFACT_ROOT = Path("docs_zh/mowa/e005_shuffled_robot_rollout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MoWA E-005 shuffled-robot rollout smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=ROLLOUT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override. When omitted, uses the checkpoint reference from config.",
    )
    parser.add_argument("--output", type=Path, default=ROLLOUT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--server-ready-timeout", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_or_plan_e005_shuffled_robot_rollout_smoke(
        args.repo_root,
        config_path=args.config,
        checkpoint_override=args.checkpoint,
        execute=args.execute,
        server_ready_timeout=args.server_ready_timeout,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def run_or_plan_e005_shuffled_robot_rollout_smoke(
    repo_root: Path | str,
    *,
    config_path: Path = ROLLOUT_CONFIG,
    checkpoint_override: Path | None = None,
    execute: bool,
    server_ready_timeout: int,
) -> dict[str, Any]:
    root = Path(repo_root)
    cfg = OmegaConf.load(root / config_path)
    checkpoint_reference = (
        checkpoint_override
        if checkpoint_override is not None
        else resolve_mowa_checkpoint_reference(
            root,
            cfg.checkpoint,
            checkpoint_root_policy=cfg.checkpoint_root_policy,
        )
    )
    cfg.checkpoint = str(checkpoint_reference)
    commands = _build_rollout_commands(cfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    artifact_dir = ARTIFACT_ROOT / stamp
    runs = []
    for intervention, command_set in commands.items():
        result = (
            _run_intervention(
                root,
                cfg,
                intervention=intervention,
                command_set=command_set,
                artifact_dir=artifact_dir,
                server_ready_timeout=server_ready_timeout,
            )
            if execute
            else {"executed": False}
        )
        runs.append(
            {
                "intervention": intervention,
                "server_command": command_set["server_command"],
                "client_command": command_set["client_command"],
                **result,
            }
        )

    checks = {
        "preflight_report_exists": (root / "docs_zh" / "mowa" / "mowa_e005_shuffled_robot_checkpoint_preflight_smoke.json").is_file(),
        "checkpoint_exists": (root / cfg.checkpoint).is_dir(),
        "batch_shuffle_batch_size_gt_1": int(cfg.client.n_envs) > 1,
        "executed_when_requested": (not execute) or all(run.get("executed") is True for run in runs),
        "server_started_when_executed": (not execute) or all(run.get("server_started") is True for run in runs),
        "client_succeeded_when_executed": (not execute) or all(
            (run.get("client_result") or {}).get("returncode") == 0 for run in runs
        ),
        "result_json_collected_when_executed": (not execute) or all(
            run.get("copied_result_json") is not None for run in runs
        ),
        "success_rate_recorded_when_executed": (not execute) or all(
            _is_number((run.get("rollout_result") or {}).get("success_rate")) for run in runs
        ),
        "structured_rollout_outcome_recorded_when_executed": (not execute)
        or bool(_extract_rollout_blocker(runs))
        or all((run.get("client_result") or {}).get("returncode") == 0 for run in runs),
    }
    rollout_blocker = _extract_rollout_blocker(runs) if execute else {}
    return {
        "stage": cfg.stage,
        "task_id": cfg.task_id,
        "experiment_id": cfg.experiment_id,
        "experiment_name": cfg.experiment_name,
        "training_started": False,
        "eval_started": bool(execute),
        "launch_ready": False,
        "execute_requested": bool(execute),
        "config": str(config_path),
        "checkpoint": str(cfg.checkpoint),
        "artifact_dir": str(artifact_dir),
        "checks": checks,
        "rollout_blocker": rollout_blocker,
        "runs": runs,
        "success_rate_delta_vs_baseline": _success_rate_deltas(runs),
        "unresolved_items": [
            "This is a tiny shuffled-robot rollout smoke, not a statistically powered benchmark.",
            "If the client or simulator fails, inspect per-intervention server/client logs under artifact_dir.",
            "Formal E-005 evidence requires more episodes/seeds after this smoke path is stable.",
        ],
        "go_no_go": _rollout_go_no_go(execute=execute, checks=checks),
    }


def _run_intervention(
    root: Path,
    cfg,
    *,
    intervention: str,
    command_set: dict[str, list[str]],
    artifact_dir: Path,
    server_ready_timeout: int,
) -> dict[str, Any]:
    run_dir = root / artifact_dir / intervention
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = run_dir / "server.log"
    client_log_path = run_dir / "client.log"
    server_log = server_log_path.open("w", encoding="utf-8")
    server = None
    start_time = time.perf_counter()
    try:
        server = subprocess.Popen(
            command_set["server_command"],
            cwd=root,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        port = _command_port(command_set["server_command"])
        server_started = _wait_for_port("127.0.0.1", port, server_ready_timeout, server)
        if not server_started:
            return {
                "executed": True,
                "server_started": False,
                "elapsed_sec": float(time.perf_counter() - start_time),
                "server_log": str(artifact_dir / intervention / "server.log"),
                "client_log": str(artifact_dir / intervention / "client.log"),
                "server_failure_category": _server_failure_category(server_log_path),
                "error": f"server port {port} did not become ready",
            }
        client_result = _run_client(root, command_set["client_command"], client_log_path)
        rollout_result, copied_result = _collect_rollout_result(
            root,
            cfg,
            artifact_dir,
            intervention,
            min_mtime=client_result["started_at_unix"],
        )
        return {
            "executed": True,
            "server_started": True,
            "elapsed_sec": float(time.perf_counter() - start_time),
            "server_log": str(artifact_dir / intervention / "server.log"),
            "client_log": str(artifact_dir / intervention / "client.log"),
            "client_result": client_result,
            "rollout_result": rollout_result,
            "copied_result_json": copied_result,
        }
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)
        server_log.close()


def _success_rate_deltas(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    baseline = _success_rate(next((run for run in runs if run["intervention"] == "baseline"), {}))
    deltas = {}
    for run in runs:
        success_rate = _success_rate(run)
        deltas[run["intervention"]] = (
            None if baseline is None or success_rate is None else float(success_rate - baseline)
        )
    return deltas


def _success_rate(run: dict[str, Any]) -> float | None:
    value = (run.get("rollout_result") or {}).get("success_rate")
    return float(value) if _is_number(value) else None


def _rollout_go_no_go(*, execute: bool, checks: dict[str, Any]) -> str:
    if not execute:
        return "TBD: E-005 rollout commands are ready for human-confirmed execution"
    return (
        "TBD: E-005 shuffled-robot rollout smoke passed; formal evidence remains gated"
        if all(checks.values())
        else "No-Go: E-005 shuffled-robot rollout smoke incomplete"
    )


if __name__ == "__main__":
    main()
