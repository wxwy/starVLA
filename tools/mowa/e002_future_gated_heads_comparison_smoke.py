"""Validate the single-entry E-002 Future-GatedHeads comparison candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


BASELINE_CONFIG = Path("configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml")
GATED_CONFIG = Path("configs/mowa/mowa_e002_future_gated_heads_candidate.yaml")
OUTPUT = Path("docs_zh/mowa/mowa_e002_future_gated_heads_comparison_smoke.json")

ALLOWED_DIFFERENCE_PATHS = {
    "launch_guard.launch_ready",
    "launch_guard.human_confirmed",
    "launch_guard.policy_confirmed",
    "launch_guard.reason",
    "run_id",
    "experiment_id",
    "inherits",
    "framework.mowa.gated_heads",
    "notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-002 Future-GatedHeads comparison smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e002_future_gated_heads_comparison_smoke(args.repo_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"go_no_go: {payload['go_no_go']}")


def build_e002_future_gated_heads_comparison_smoke(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    baseline = OmegaConf.load(root / BASELINE_CONFIG)
    gated_overlay = OmegaConf.load(root / GATED_CONFIG)
    gated = OmegaConf.merge(baseline, gated_overlay)
    differences = _diff_configs(
        OmegaConf.to_container(baseline, resolve=True),
        OmegaConf.to_container(gated, resolve=True),
    )
    unexpected = sorted(path for path in differences if not _is_allowed_difference(path))
    gated_heads_cfg = OmegaConf.to_container(
        OmegaConf.select(gated, "framework.mowa.gated_heads"),
        resolve=True,
    )
    checks = {
        "baseline_config_exists": (root / BASELINE_CONFIG).is_file(),
        "gated_config_exists": (root / GATED_CONFIG).is_file(),
        "gated_candidate_references_fullheads_control": OmegaConf.select(
            gated,
            "framework.mowa.gated_heads.fullheads_control_config",
        )
        == str(BASELINE_CONFIG),
        "single_fullheads_comparison_only": OmegaConf.select(
            gated,
            "framework.mowa.gated_heads.comparison_scope",
        )
        == "single_fullheads_control_only",
        "per_head_sweep_disabled": OmegaConf.select(
            gated,
            "framework.mowa.gated_heads.allow_per_head_sweep",
        )
        is False,
        "gated_heads_enabled": OmegaConf.select(gated, "framework.mowa.gated_heads.enabled") is True,
        "candidate_is_launch_gated": OmegaConf.select(gated, "launch_guard.launch_ready") is False,
        "unexpected_difference_paths_empty": not unexpected,
    }
    return {
        "task_id": "M2-003",
        "experiment_id": "E-002",
        "training_started": False,
        "checks": checks,
        "configs": {
            "baseline_fullheads": str(BASELINE_CONFIG),
            "gated_heads_candidate": str(GATED_CONFIG),
        },
        "observed": {
            "difference_paths": sorted(differences),
            "unexpected_difference_paths": unexpected,
            "gated_heads": gated_heads_cfg,
        },
        "unresolved_items": [
            "This is the single static E-002 comparison entry only; "
            "runtime integration exists but training remains gated.",
            "Per-head / leave-one-out / selected-head sweeps remain forbidden.",
            "No training is started by this smoke.",
        ],
        "go_no_go": (
            "TBD: E-002 single FullHeads comparison entry is wired; training remains gated"
            if all(checks.values())
            else "No-Go: E-002 comparison candidate drift detected"
        ),
    }


def _diff_configs(left: Any, right: Any, prefix: str = "") -> set[str]:
    differences: set[str] = set()
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        for key in sorted(keys):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.add(path)
                continue
            differences |= _diff_configs(left[key], right[key], path)
        return differences
    if left != right:
        differences.add(prefix)
    return differences


def _is_allowed_difference(path: str) -> bool:
    return any(path == allowed or path.startswith(f"{allowed}.") for allowed in ALLOWED_DIFFERENCE_PATHS)


if __name__ == "__main__":
    main()
