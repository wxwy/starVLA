"""Run the E-004 HLC-GCI checkpoint-backed policy rollout smoke."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    from tools.mowa.mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference


ROLLOUT_CONFIG = Path("configs/mowa/smoke/mowa_e004_hlc_gci_policy_rollout_candidate.yaml")
ROLLOUT_OUTPUT = Path("docs_zh/mowa/mowa_e004_hlc_gci_policy_rollout_smoke.json")
ARTIFACT_ROOT = Path("docs_zh/mowa/e004_hlc_gci_policy_rollout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MoWA E-004 HLC-GCI policy rollout smoke.")
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
    payload = run_or_plan_e004_hlc_gci_policy_rollout_smoke(
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


def run_or_plan_e004_hlc_gci_policy_rollout_smoke(
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
        if execute and (
            result.get("server_started") is False
            or (result.get("client_result") or {}).get("returncode") not in (None, 0)
        ):
            break

    checks = {
        "preflight_report_exists": (root / "docs_zh" / "mowa" / "mowa_e004_hlc_gci_checkpoint_preflight_smoke.json").is_file(),
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
            "This is a tiny RoboCasa365 rollout smoke, not a statistically powered benchmark.",
            "If the client or simulator fails, inspect per-intervention server/client logs under artifact_dir.",
            "Formal E-004 evidence requires more episodes/seeds after this smoke path is stable.",
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


def _run_client(root: Path, command: list[str], log_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    tail = _tail_lines(log_path)
    return {
        "returncode": result.returncode,
        "elapsed_sec": float(time.perf_counter() - start),
        "started_at_unix": started_at,
        "tail": tail,
        "failure_category": _client_failure_category(tail) if result.returncode != 0 else None,
    }


def _build_rollout_commands(cfg) -> dict[str, dict[str, list[str]]]:
    commands = {}
    for index, intervention in enumerate(cfg.interventions):
        port = int(cfg.server.port_base) + index
        override = f"framework.mowa.layerwise_bridge_token_intervention={intervention}"
        server_command = [
            str(cfg.server.python),
            str(cfg.server.entrypoint),
            "--ckpt_path",
            str(cfg.checkpoint),
            "--port",
            str(port),
            "--idle_timeout",
            str(cfg.server.idle_timeout),
            "--config_override",
            override,
        ]
        if bool(cfg.server.use_bf16):
            server_command.append("--use_bf16")
        client_command = [
            str(cfg.client.python),
            "-m",
            str(cfg.client.module),
            "--args.pretrained-path",
            str(cfg.checkpoint),
            "--args.env-name",
            str(cfg.client.env_name),
            "--args.port",
            str(port),
            "--args.n-episodes",
            str(cfg.client.n_episodes),
            "--args.n-envs",
            str(cfg.client.n_envs),
            "--args.max-episode-steps",
            str(cfg.client.max_episode_steps),
            "--args.n-action-steps",
            str(cfg.client.n_action_steps),
            "--args.video-out-path",
            str(Path(cfg.client.video_out_path) / intervention),
        ]
        commands[intervention] = {
            "server_command": server_command,
            "client_command": client_command,
        }
    return commands


def _wait_for_port(host: str, port: int, timeout: int, process: subprocess.Popen) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def _command_port(command: list[str]) -> int:
    port_index = command.index("--port") + 1
    return int(command[port_index])


def _collect_rollout_result(
    root: Path,
    cfg,
    artifact_dir: Path,
    intervention: str,
    *,
    min_mtime: float,
) -> tuple[dict[str, Any], str | None]:
    result_path = (root / cfg.checkpoint).with_suffix(".eval") / str(cfg.client.env_name).replace("/", "_")
    result_path = result_path.with_suffix(".json")
    if not result_path.is_file():
        return {}, None
    if result_path.stat().st_mtime < min_mtime:
        return {"stale_result_path": str(result_path)}, None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    copied = artifact_dir / intervention / result_path.name
    copied_abs = root / copied
    copied_abs.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_path, copied_abs)
    return payload, str(copied)


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tail_lines(path: Path, limit: int = 60) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]


def _client_failure_category(lines: list[str]) -> str:
    text = "\n".join(lines)
    if "No module named" in text:
        return "missing_python_dependency"
    if (
        "mujoco.osmesa" in text and "glGetError" in text
    ) or (
        "mujoco.egl" in text and "eglQueryString" in text
    ):
        return "robocasa_render_backend_unavailable"
    if "FileNotFoundError" in text and "models/assets" in text:
        return "missing_robocasa_asset"
    if "ConnectionRefusedError" in text or "Failed to connect to server" in text:
        return "policy_server_connection_failed"
    return "client_failed"


def _rollout_go_no_go(*, execute: bool, checks: dict[str, bool]) -> str:
    if execute and checks.get("structured_rollout_outcome_recorded_when_executed") and not checks.get(
        "client_succeeded_when_executed"
    ):
        return "No-Go: E-004 rollout blocked; see rollout_blocker"
    if not all(checks.values()):
        return "No-Go: E-004 rollout smoke incomplete"
    if not execute:
        return "TBD: E-004 rollout smoke plan ready; execution not started"
    return "TBD: E-004 rollout smoke executed; review success deltas before any claim"


def _extract_rollout_blocker(runs: list[dict[str, Any]]) -> dict[str, Any]:
    server_blocker = _extract_server_blocker(runs)
    if server_blocker:
        return server_blocker
    render_blocker = _extract_render_backend_blocker(runs)
    if render_blocker:
        return render_blocker
    asset_blocker = _extract_asset_blocker(runs)
    if asset_blocker:
        return asset_blocker
    return {}


def _extract_server_blocker(runs: list[dict[str, Any]]) -> dict[str, Any]:
    failing_runs = [
        run
        for run in runs
        if run.get("server_failure_category") == "checkpoint_model_incompatible"
    ]
    if not failing_runs:
        return {}
    return {
        "status": "checkpoint_model_incompatible",
        "scope": "checkpoint",
        "blocked_interventions": [run.get("intervention") for run in failing_runs],
        "missing_state_keys": _collect_server_missing_keys(failing_runs),
    }


def _extract_asset_blocker(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    failing_runs = [
        run
        for run in runs
        if (run.get("client_result") or {}).get("failure_category") == "missing_robocasa_asset"
    ]
    if not failing_runs:
        return {}
    missing_assets = []
    interventions = []
    for run in failing_runs:
        interventions.append(run.get("intervention"))
        tail = (run.get("client_result") or {}).get("tail") or []
        path = _extract_missing_asset_path(tail)
        if path is not None and path not in missing_assets:
            missing_assets.append(path)
    return {
        "status": "missing_robocasa_assets",
        "scope": "environment",
        "blocked_interventions": interventions,
        "missing_asset_paths": missing_assets,
    }


def _extract_render_backend_blocker(runs: list[dict[str, Any]]) -> dict[str, Any]:
    failing_runs = [
        run
        for run in runs
        if (run.get("client_result") or {}).get("failure_category") == "robocasa_render_backend_unavailable"
    ]
    if not failing_runs:
        return {}
    return {
        "status": "robocasa_render_backend_unavailable",
        "scope": "environment",
        "blocked_interventions": [run.get("intervention") for run in failing_runs],
        "backend_signatures": _collect_render_backend_signatures(failing_runs),
    }


def _server_failure_category(path: Path) -> str | None:
    lines = _tail_lines(path)
    text = "\n".join(lines)
    if "Error(s) in loading state_dict" in text and "Missing key(s) in state_dict" in text:
        return "checkpoint_model_incompatible"
    return None


def _collect_server_missing_keys(runs: list[dict[str, Any]]) -> list[str]:
    keys = []
    for run in runs:
        for line in _tail_lines(Path(run["server_log"])):
            if "Missing key(s) in state_dict:" not in line:
                continue
            _, _, suffix = line.partition("Missing key(s) in state_dict:")
            for raw_key in suffix.split(","):
                key = raw_key.strip().strip("[]'\"")
                if key and key not in keys:
                    keys.append(key)
    return keys


def _collect_render_backend_signatures(runs: list[dict[str, Any]]) -> list[str]:
    signatures = []
    for run in runs:
        for line in (run.get("client_result") or {}).get("tail") or []:
            if "glGetError" in line or "eglQueryString" in line:
                if line not in signatures:
                    signatures.append(line)
    return signatures


def _extract_missing_asset_path(lines: list[str]) -> str | None:
    for line in reversed(lines):
        if "FileNotFoundError" not in line or "models/assets" not in line:
            continue
        marker = "No such file or directory: "
        if marker not in line:
            continue
        return line.split(marker, 1)[1].strip().strip("'\"")
    return None


if __name__ == "__main__":
    main()
