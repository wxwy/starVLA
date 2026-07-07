"""Run OpenDrawer simulator-state mapping audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import build_mowa_opendrawer_state_mapping_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit OpenDrawer states.npz to drawer-joint mapping.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot"),
        help="OpenDrawer LeRobot dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. When omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_mowa_opendrawer_state_mapping_audit(args.dataset_path).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
