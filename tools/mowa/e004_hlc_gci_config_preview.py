"""Preview MoWA E-004 HLC-GCI config before any rollout work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from starVLA.dataloader.mowa import DATA_GATE


CONFIG = Path("configs/mowa/mowa_e004_hlc_gci_candidate.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-004 config preview.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config-yaml", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e004_hlc_gci_config_preview(args.repo_root, args.config_yaml)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e004_hlc_gci_config_preview(
    repo_root: Path | str,
    config_yaml: Path | str = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_path = root / Path(config_yaml)
    cfg = _load_yaml(config_path)
    interface = _select(cfg, "interface") or {}
    history_contract = _select(cfg, "history_contract") or {}
    status = _select(cfg, "status") or {}
    references = _select(cfg, "references") or {}
    resolved_final_values = {
        "project_short_name": _select(cfg, "project_short_name"),
        "stage": _select(cfg, "stage"),
        "experiment_id": _select(cfg, "experiment_id"),
        "config_role": _select(cfg, "config_role"),
        "launch_ready": _select(cfg, "launch_guard.launch_ready"),
        "requires_human_confirmation": _select(cfg, "launch_guard.requires_human_confirmation"),
        "human_confirmed": _select(cfg, "launch_guard.human_confirmed"),
        "policy_confirmed": _select(cfg, "launch_guard.policy_confirmed"),
        "history_latent_dim": interface.get("history_latent_dim"),
        "condition_hidden_dim": interface.get("condition_hidden_dim"),
        "history_steps": interface.get("history_steps"),
        "compressed_history_dim": interface.get("compressed_history_dim"),
        "gate_hidden_dim": interface.get("gate_hidden_dim"),
        "batch_size_smoke": interface.get("batch_size_smoke"),
        "condition_token_count": interface.get("condition_token_count"),
        "injection_policy": interface.get("injection_policy"),
        "sampling_policy": history_contract.get("sampling_policy"),
        "history_source": history_contract.get("history_source"),
        "future_action_policy": history_contract.get("future_action_policy"),
        "leakage_guard": history_contract.get("leakage_guard"),
        "history_sampling_status": status.get("history_sampling_status", DATA_GATE),
        "gate_init_status": status.get("gate_init_status", DATA_GATE),
        "shape_status": status.get("shape_status", DATA_GATE),
        "reference_count": len(references),
    }
    checks = {
        "config_created": config_path.is_file(),
        "launch_guard_closed": _select(cfg, "launch_guard.launch_ready") is False,
        "injection_policy_condition_path_only": interface.get("injection_policy")
        == "condition_path_only",
        "history_source_robot_only": history_contract.get("history_source")
        == "robot_history_latent_only",
        "future_action_policy_target_only": history_contract.get("future_action_policy") == "target_only",
        "leakage_guard_future_action_not_input": history_contract.get("leakage_guard")
        == "future_action_not_input",
        "history_sampling_status_data_gate": status.get("history_sampling_status", DATA_GATE)
        == DATA_GATE,
        "gate_init_status_data_gate": status.get("gate_init_status", DATA_GATE) == DATA_GATE,
        "shape_status_data_gate": status.get("shape_status", DATA_GATE) == DATA_GATE,
        "references_present": bool(references),
    }
    return {
        "stage": "hlc_gci",
        "task_id": "M4-001",
        "experiment_id": "E-004",
        "config_role": "hlcgci_candidate",
        "launch_ready": bool(_select(cfg, "launch_guard.launch_ready")),
        "config_path": str(Path(config_yaml)),
        "resolved_final_values": resolved_final_values,
        "checks": checks,
        "unresolved_items": [
            "This preview validates E-004 interface shape and history contract only.",
            "Shuffled-robot metrics and framework injection still require a meaningful checkpoint.",
        ],
        "go_no_go": (
            "TBD: E-004 config preview passed; rollout remains gated"
            if all(checks.values())
            else "No-Go: E-004 config preview incomplete"
        ),
    }


def _load_yaml(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return OmegaConf.load(path)


def _select(cfg: Any | None, dot_path: str) -> Any:
    if cfg is None:
        return None
    value = OmegaConf.select(cfg, dot_path, default=None)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


if __name__ == "__main__":
    main()
