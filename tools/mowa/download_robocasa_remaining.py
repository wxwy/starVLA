"""Resume the optional RoboCasa365 target/composite downloads.

The upstream robocasa download_datasets.py uses urllib and has been failing
with SSL UNEXPECTED_EOF_WHILE_READING errors. This script uses wget with
resume support (-c) and curl as a fallback, then extracts the tar and removes
it, matching the layout expected by robocasa.

This is an optional data maintenance helper. By default it prints the planned
downloads only; pass ``--execute`` to perform network downloads and extraction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import robocasa
from robocasa.macros import DATASET_BASE_PATH


BOX_LINKS_DS_PATH = Path(robocasa.__path__[0]) / "models" / "assets" / "box_links" / "box_links_ds.json"

REMAINING_TASKS = [
    ("SeparateFreezerRack", "20250815"),
    ("SetUpCuttingStation", "20250817"),
    ("StackBowlsCabinet", "20250815"),
    ("SteamInMicrowave", "20250814"),
    ("StirVegetables", "20250814"),
    ("StoreLeftoversInBowl", "20250813"),
    ("WaffleReheat", "20250817"),
    ("WashFruitColander", "20250811"),
    ("WashLettuce", "20250814"),
    ("WeighIngredients", "20250812"),
]


def _get_direct_download_url(shared_url: str, ext: str = "tar") -> str:
    shared_id = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{shared_id}.{ext}"


def _download_with_wget(url: str, output_path: Path, timeout: int = 600) -> bool:
    """Try wget with resume. Return True on success."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "-c",  # continue partial downloads
        "--tries=5",
        "--timeout=60",
        "--progress=dot:giga",
        "-O", str(output_path),
        url,
    ]
    print(f"[wget] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, timeout=timeout, check=False)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except subprocess.TimeoutExpired:
        print(f"[wget] timed out after {timeout}s")
        return False


def _download_with_curl(url: str, output_path: Path, timeout: int = 600) -> bool:
    """Fallback to curl with resume. Return True on success."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-C", "-",  # resume
        "--retry", "5",
        "--retry-delay", "5",
        "--max-time", str(timeout),
        "-L",  # follow redirects
        "-o", str(output_path),
        url,
    ]
    print(f"[curl] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, timeout=timeout + 30, check=False)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except subprocess.TimeoutExpired:
        print(f"[curl] timed out after {timeout + 30}s")
        return False


def _extract_and_cleanup(tar_path: Path, extract_dir: Path) -> bool:
    print(f"[extract] {tar_path} -> {extract_dir}")
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=str(extract_dir))
        tar_path.unlink()
        print(f"[cleanup] removed {tar_path}")
        return True
    except Exception as exc:
        print(f"[extract] failed: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume optional RoboCasa365 target/composite downloads.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform network downloads and extraction. Default only prints the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with BOX_LINKS_DS_PATH.open("r", encoding="utf-8") as f:
        box_links = json.load(f)

    if DATASET_BASE_PATH is not None:
        base_datasets_path = Path(DATASET_BASE_PATH) / "v1.0"
    else:
        base_datasets_path = Path(robocasa.__path__[0]).parent / "datasets" / "v1.0"

    failed: list[str] = []
    succeeded: list[str] = []

    for task_name, date in REMAINING_TASKS:
        tar_key = f"target/composite/{task_name}/{date}/lerobot.tar"
        shared_url = box_links.get(tar_key)
        if shared_url is None:
            print(f"[skip] no Box link for {tar_key}")
            failed.append(task_name)
            continue

        url = _get_direct_download_url(shared_url, ext="tar")
        ds_path = base_datasets_path / f"target/composite/{task_name}/{date}/lerobot"
        extract_dir = ds_path.parent
        tar_path = extract_dir / "lerobot.tar"
        if not args.execute:
            status = "present" if (ds_path / "meta" / "episodes.jsonl").exists() else "missing"
            print(f"[plan] {task_name}: status={status}, destination={tar_path}")
            continue

        # Skip if already extracted.
        if (ds_path / "meta" / "episodes.jsonl").exists() and (ds_path / "data").is_dir():
            print(f"[skip] {task_name} already extracted")
            succeeded.append(task_name)
            continue

        print(f"\n=== {task_name} ({tar_key}) ===")
        print(f"[info] destination: {tar_path}")

        # Download if not already complete.
        if not tar_path.exists() or tar_path.stat().st_size == 0:
            ok = _download_with_wget(url, tar_path)
            if not ok:
                ok = _download_with_curl(url, tar_path)
        else:
            print(f"[resume] partial tar exists ({tar_path.stat().st_size} bytes), attempting resume")
            ok = _download_with_wget(url, tar_path)
            if not ok:
                ok = _download_with_curl(url, tar_path)

        if not ok:
            print(f"[fail] download failed for {task_name}")
            failed.append(task_name)
            continue

        # Extract.
        if _extract_and_cleanup(tar_path, extract_dir):
            succeeded.append(task_name)
        else:
            failed.append(task_name)

    if not args.execute:
        print("\n=== Plan only ===")
        print("Pass --execute to download and extract missing target/composite tasks.")
        return 0

    print("\n=== Summary ===")
    print(f"Succeeded: {len(succeeded)}/{len(REMAINING_TASKS)}")
    for name in succeeded:
        print(f"  ✓ {name}")
    if failed:
        print(f"Failed: {len(failed)}/{len(REMAINING_TASKS)}")
        for name in failed:
            print(f"  ✗ {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
