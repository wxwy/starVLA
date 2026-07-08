"""Run the E-004 HLC-GCI launch smoke."""

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
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from e004_hlc_gci_config_preview import build_e004_hlc_gci_config_preview  # type: ignore[no-redef]
    from hlc_gci_interface_smoke import build_hlc_gci_interface_smoke  # type: ignore[no-redef]
from starVLA.model.modules.mowa import MoWAHLCGCI, MoWAHLCGCIConfig


CONFIG = Path("configs/mowa/mowa_e004_hlc_gci_launch_candidate.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MoWA E-004 launch smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config-yaml", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e004_hlc_gci_launch_smoke(args.repo_root, args.config_yaml)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e004_hlc_gci_launch_smoke(
    repo_root: Path | str,
    config_yaml: Path | str = CONFIG,
) -> dict[str, Any]:
    root = Path(repo_root)
    config_path = root / Path(config_yaml)
    cfg = _load_yaml(config_path)
    launch_guard = _select(cfg, "launch_guard") or {}
    interface = _select(cfg, "interface") or {}
    smoke_cfg = _select(cfg, "smoke") or {}

    preview_report = build_e004_hlc_gci_config_preview(root)
    interface_report = build_hlc_gci_interface_smoke(root)
    synthetic_train_report = _run_synthetic_train_smoke(interface, smoke_cfg)

    checks = {
        "config_created": config_path.is_file(),
        "launch_guard_open": bool(launch_guard.get("launch_ready")) is True,
        "launch_guard_human_confirmed": bool(launch_guard.get("human_confirmed")) is True,
        "launch_guard_policy_confirmed": bool(launch_guard.get("policy_confirmed")) is True,
        "preview_passed": _report_is_ok(preview_report),
        "interface_passed": _report_is_ok(interface_report),
        "synthetic_train_passed": _report_is_ok(synthetic_train_report),
    }
    return {
        "stage": "hlc_gci",
        "task_id": "M4-001",
        "experiment_id": "E-004",
        "training_started": False,
        "launch_ready": bool(launch_guard.get("launch_ready")),
        "checks": checks,
        "config_path": str(Path(config_yaml)),
        "reports": {
            "config_preview": preview_report,
            "interface_smoke": interface_report,
            "synthetic_train_smoke": synthetic_train_report,
        },
        "observed": {
            "history_latent_dim": interface.get("history_latent_dim"),
            "condition_hidden_dim": interface.get("condition_hidden_dim"),
            "history_steps": interface.get("history_steps"),
            "compressed_history_dim": interface.get("compressed_history_dim"),
            "gate_hidden_dim": interface.get("gate_hidden_dim"),
        },
        "unresolved_items": [
            "This launch smoke validates the interface, preview, and a synthetic optimization loop.",
            "Real shuffled-robot rollout still requires a meaningful checkpoint and checkpoint-backed policy path.",
        ],
        "go_no_go": (
            "TBD: E-004 launch smoke passed; rollout remains gated"
            if all(checks.values())
            else "No-Go: E-004 launch smoke incomplete"
        ),
    }


def _run_synthetic_train_smoke(interface: dict[str, Any], smoke_cfg: dict[str, Any]) -> dict[str, Any]:
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(smoke_cfg.get("learning_rate", 1.0e-4)))
    generator = torch.Generator().manual_seed(0)
    history_latent = torch.randn(
        batch_size,
        history_steps,
        history_latent_dim,
        generator=generator,
    )
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
        "go_no_go": "TBD: E-004 synthetic train smoke passed; rollout remains gated",
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


if __name__ == "__main__":
    main()
