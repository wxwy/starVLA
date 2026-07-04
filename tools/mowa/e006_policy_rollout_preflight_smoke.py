"""Validate MoWA E-006 RoboCasa365 rollout command candidate without launching eval."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


CONFIG = Path("configs/mowa/mowa_e006_policy_rollout_candidate.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_e006_policy_rollout_preflight_smoke.json")
RUN_EVAL = Path("examples/Robocasa_365/eval_files/run_eval.sh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-006 policy rollout preflight smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e006_policy_rollout_preflight_smoke(args.repo_root, args.config)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e006_policy_rollout_preflight_smoke(
    repo_root: Path | str,
    config_path: Path = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_file = root / config_path
    cfg = OmegaConf.load(config_file)
    commands = _build_rollout_commands(cfg)
    bash_syntax = subprocess.run(
        ["bash", "-n", str(RUN_EVAL)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    checks = {
        "config_exists": config_file.is_file(),
        "launch_ready_false": cfg.get("launch_ready") is False,
        "eval_started_false": cfg.get("eval_started") is False,
        "requires_human_confirmation": cfg.get("requires_human_confirmation") is True,
        "checkpoint_exists": (root / cfg.checkpoint).is_dir(),
        "checkpoint_under_mowa_ckpt": str(cfg.checkpoint).startswith(str(cfg.checkpoint_root_policy)),
        "server_entrypoint_exists": (root / cfg.server.entrypoint).is_file(),
        "client_module_exists": (root / Path(str(cfg.client.module).replace(".", "/")).with_suffix(".py")).is_file(),
        "starvla_python_exists": (root / cfg.server.python).is_file(),
        "robocasa_python_exists": (root / cfg.client.python).is_file(),
        "run_eval_shell_syntax_valid": bash_syntax.returncode == 0,
        "all_interventions_have_commands": set(commands.keys()) == set(cfg.interventions),
        "non_baseline_commands_use_config_override": all(
            "--config_override" in command_set["server_command"]
            for name, command_set in commands.items()
            if name != "baseline"
        ),
        "baseline_command_pins_baseline_override": (
            "framework.mowa.layerwise_bridge_token_intervention=baseline"
            in commands["baseline"]["server_command"]
        ),
        "preflight_does_not_launch_eval": True,
    }
    return {
        "stage": cfg.stage,
        "task_id": cfg.task_id,
        "experiment_id": cfg.experiment_id,
        "experiment_name": cfg.experiment_name,
        "training_started": False,
        "eval_started": False,
        "launch_ready": False,
        "checks": checks,
        "config": str(config_path),
        "commands": commands,
        "unresolved_items": [
            "This preflight validates rollout commands only; it does not start the policy server or RoboCasa client.",
            "True E-006 action success deltas require running server/client for each intervention.",
            "The server-side config override path is default-off and only active when --config_override is provided.",
        ],
        "go_no_go": (
            "TBD: E-006 rollout commands are ready for human-confirmed execution"
            if all(checks.values())
            else "No-Go: E-006 rollout preflight failed"
        ),
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


if __name__ == "__main__":
    main()
