"""Run the E-004 HLC-GCI checkpoint-backed preflight smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

try:
    from tools.mowa.e004_hlc_gci_config_preview import build_e004_hlc_gci_config_preview
    from tools.mowa.hlc_gci_interface_smoke import build_hlc_gci_interface_smoke
    from tools.mowa.mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from e004_hlc_gci_config_preview import build_e004_hlc_gci_config_preview  # type: ignore[no-redef]
    from hlc_gci_interface_smoke import build_hlc_gci_interface_smoke  # type: ignore[no-redef]
    from mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference  # type: ignore[no-redef]
from starVLA.model.modules.mowa import MoWAHLCGCI, MoWAHLCGCIConfig


CONFIG = Path("configs/mowa/mowa_e004_hlc_gci_checkpoint_candidate.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_e004_hlc_gci_checkpoint_preflight_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MoWA E-004 checkpoint preflight smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config-yaml", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e004_hlc_gci_checkpoint_preflight_smoke(
        args.repo_root,
        args.config_yaml,
        checkpoint=args.checkpoint,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e004_hlc_gci_checkpoint_preflight_smoke(
    repo_root: Path | str,
    config_yaml: Path | str = CONFIG,
    *,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_path = root / Path(config_yaml)
    cfg = _load_yaml(config_path)
    interface = _select(cfg, "interface") or {}
    checkpoint_cfg = _select(cfg, "checkpoint") or {}
    resolved_checkpoint = checkpoint or _resolve_checkpoint(root, checkpoint_cfg)
    checkpoint_path = resolved_checkpoint if resolved_checkpoint.is_absolute() else root / resolved_checkpoint

    preview_report = build_e004_hlc_gci_config_preview(root)
    interface_report = build_hlc_gci_interface_smoke(root)
    checkpoint_report = _inspect_checkpoint(checkpoint_path, root)
    synthetic_train_report = _run_synthetic_train_smoke(interface)

    checks = {
        "config_created": config_path.is_file(),
        "checkpoint_exists": checkpoint_path.is_dir(),
        "checkpoint_trainer_state_present": checkpoint_report["trainer_state_present"],
        "checkpoint_completed_steps_positive": checkpoint_report["completed_steps_positive"],
        "preview_passed": _report_is_ok(preview_report),
        "interface_passed": _report_is_ok(interface_report),
        "synthetic_train_passed": _report_is_ok(synthetic_train_report),
    }
    return {
        "stage": "hlc_gci",
        "task_id": "M4-001",
        "experiment_id": "E-004",
        "training_started": False,
        "launch_ready": False,
        "checks": checks,
        "config_path": str(config_path.relative_to(root) if config_path.is_relative_to(root) else config_path),
        "observed": {
            "checkpoint": str(checkpoint_path.relative_to(root) if checkpoint_path.is_relative_to(root) else checkpoint_path),
            "checkpoint_source_hint": checkpoint_cfg.get("checkpoint_source_hint"),
            "checkpoint_root_policy": checkpoint_cfg.get("checkpoint_root_policy"),
            "checkpoint_completed_steps": checkpoint_report["completed_steps"],
            "checkpoint_under_policy_root": checkpoint_report["under_policy_root"],
            "history_latent_dim": interface.get("history_latent_dim"),
            "condition_hidden_dim": interface.get("condition_hidden_dim"),
            "history_steps": interface.get("history_steps"),
            "compressed_history_dim": interface.get("compressed_history_dim"),
            "gate_hidden_dim": interface.get("gate_hidden_dim"),
        },
        "reports": {
            "config_preview": preview_report,
            "interface_smoke": interface_report,
            "checkpoint_report": checkpoint_report,
            "synthetic_train_smoke": synthetic_train_report,
        },
        "unresolved_items": [
            "This preflight validates checkpoint-backed wiring and HLC-GCI smoke only; it does not run shuffled-robot rollout.",
            "The checkpoint used here is an upstream complete checkpoint reference, not an E-004-owned rollout checkpoint.",
        ],
        "go_no_go": (
            "TBD: E-004 checkpoint-backed preflight passed; rollout remains gated"
            if all(checks.values())
            else "No-Go: E-004 checkpoint-backed preflight incomplete"
        ),
    }


def _inspect_checkpoint(checkpoint_path: Path, root: Path) -> dict[str, Any]:
    trainer_state = _read_json(checkpoint_path / "trainer_state.json") or {}
    completed_steps = trainer_state.get("completed_steps")
    under_policy_root = str(checkpoint_path).startswith(str(root / "playground/mowa_ckpt"))
    return {
        "trainer_state_present": (checkpoint_path / "trainer_state.json").is_file(),
        "completed_steps": completed_steps,
        "completed_steps_positive": isinstance(completed_steps, int) and completed_steps > 0,
        "under_policy_root": under_policy_root,
    }


def _run_synthetic_train_smoke(interface: dict[str, Any]) -> dict[str, Any]:
    batch_size = int(interface.get("batch_size_smoke", 2))
    history_steps = int(interface.get("history_steps", 10))
    history_latent_dim = int(interface.get("history_latent_dim", 1024))
    condition_hidden_dim = int(interface.get("condition_hidden_dim", 1024))
    compressed_history_dim = int(interface.get("compressed_history_dim", 512))
    gate_hidden_dim = int(interface.get("gate_hidden_dim", 256))
    model = MoWAHLCGCI(
        MoWAHLCGCIConfig(
            history_latent_dim=history_latent_dim,
            condition_hidden_dim=condition_hidden_dim,
            history_steps=history_steps,
            compressed_history_dim=compressed_history_dim,
            gate_hidden_dim=gate_hidden_dim,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    generator = torch.Generator().manual_seed(7)
    history_latent = torch.randn(batch_size, history_steps, history_latent_dim, generator=generator)
    condition_tokens = torch.randn(
        batch_size,
        int(interface.get("condition_token_count", 4)),
        condition_hidden_dim,
        generator=generator,
    )
    output = model(history_latent, condition_tokens)
    loss = output.gate_values.mean() + output.compressed_history.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "go_no_go": "TBD: E-004 synthetic checkpoint-backed smoke passed; rollout remains gated",
        "checks": {
            "model_forward_succeeds": tuple(output.compressed_history.shape) == (batch_size, compressed_history_dim),
            "loss_is_finite": bool(torch.isfinite(loss).item()),
            "optimizer_step_succeeds": True,
        },
        "observed": {
            "compressed_history_shape": tuple(output.compressed_history.shape),
            "gate_shape": tuple(output.gate_values.shape),
            "gated_condition_shape": tuple(output.gated_condition_tokens.shape),
            "loss": float(loss.item()),
        },
    }


def _resolve_checkpoint(root: Path, checkpoint_cfg: Any) -> Path:
    checkpoint_root_policy = Path(str((checkpoint_cfg or {}).get("checkpoint_root_policy", "playground/mowa_ckpt")))
    return resolve_mowa_checkpoint_reference(
        root,
        (checkpoint_cfg or {}).get("checkpoint", "latest_complete"),
        checkpoint_root_policy=checkpoint_root_policy,
    )


def _report_is_ok(report: dict[str, Any]) -> bool:
    if not report:
        return False
    if report.get("go_no_go", "").startswith("No-Go"):
        return False
    checks = report.get("checks") or {}
    return bool(checks) and all(bool(value) for value in checks.values())


def _load_yaml(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return OmegaConf.load(path)


def _select(cfg: Any | None, dot_path: str) -> Any:
    if cfg is None:
        return None
    value = OmegaConf.select(cfg, dot_path, default=None)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
