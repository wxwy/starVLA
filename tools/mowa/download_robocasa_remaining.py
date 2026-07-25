"""RoboCasa365 dataset download helper.

Supports configurable subsets with resume and extraction. Default behavior is
plan-only for the current configured subset (10 remaining target/composite
tasks). Pass --execute to download/extract, or pass --subset/--tasks/--split/
--source to select other data.

Examples:
    # Preview current configuration (default)
    python tools/mowa/download_robocasa_remaining.py

    # Finish current configuration
    python tools/mowa/download_robocasa_remaining.py --execute

    # All target composite tasks
    python tools/mowa/download_robocasa_remaining.py --subset target/composite --execute

    # All target atomic tasks
    python tools/mowa/download_robocasa_remaining.py --subset target/atomic

    # Pretrain atomic + composite
    python tools/mowa/download_robocasa_remaining.py --subset pretrain

    # Specific tasks (dates resolved automatically)
    python tools/mowa/download_robocasa_remaining.py --tasks OpenDrawer TurnOnSinkFaucet

    # Human-only, target-only subset
    python tools/mowa/download_robocasa_remaining.py --subset target --source human
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

# Current configuration: the 10 remaining target/composite tasks being downloaded.
CURRENT_CONFIG_TASKS: list[tuple[str, str]] = [
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

SUBSET_CHOICES = [
    "current",
    "target",
    "pretrain",
    "target/atomic",
    "target/composite",
    "pretrain/atomic",
    "pretrain/composite",
    "all",
]


def _get_direct_download_url(shared_url: str, ext: str = "tar") -> str:
    """Convert a Box shared URL to a direct-download static URL."""
    shared_id = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{shared_id}.{ext}"


def _dataset_base_path() -> Path:
    """Resolve the robocasa datasets root path."""
    if DATASET_BASE_PATH is not None:
        return Path(DATASET_BASE_PATH) / "v1.0"
    return Path(robocasa.__path__[0]).parent / "datasets" / "v1.0"


def _load_box_links() -> dict[str, str]:
    with BOX_LINKS_DS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_extracted(ds_path: Path) -> bool:
    """Check whether a dataset has been downloaded and extracted."""
    return (
        ds_path.is_dir()
        and (ds_path / "meta" / "episodes.jsonl").is_file()
        and (ds_path / "data").is_dir()
    )


def _download_with_wget(url: str, output_path: Path, timeout: int) -> bool:
    """Download or resume via wget. Returns True when file is complete."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "-c",  # resume partial downloads
        "--tries=10",
        "--timeout=60",
        "--progress=dot:giga",
        "-O",
        str(output_path),
        url,
    ]
    print(f"[wget] {url}\n       -> {output_path}")
    try:
        result = subprocess.run(cmd, timeout=timeout, check=False)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except subprocess.TimeoutExpired:
        print(f"[wget] timed out after {timeout}s")
        return False


def _download_with_curl(url: str, output_path: Path, timeout: int) -> bool:
    """Fallback download or resume via curl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-C",
        "-",  # resume
        "--retry",
        "10",
        "--retry-delay",
        "5",
        "--max-time",
        str(timeout),
        "-L",
        "-o",
        str(output_path),
        url,
    ]
    print(f"[curl] {url}\n       -> {output_path}")
    try:
        result = subprocess.run(cmd, timeout=timeout + 30, check=False)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except subprocess.TimeoutExpired:
        print(f"[curl] timed out after {timeout + 30}s")
        return False


def _extract_and_cleanup(tar_path: Path, extract_dir: Path, *, keep_tar: bool = False) -> bool:
    """Extract a tar archive and remove it on success."""
    print(f"[extract] {tar_path} -> {extract_dir}")
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=str(extract_dir))
        if keep_tar:
            print(f"[cleanup] kept {tar_path}")
        else:
            print(f"[cleanup] removed {tar_path}")
            tar_path.unlink()
        return True
    except Exception as exc:
        print(f"[extract] failed: {exc}")
        return False


def _parse_tar_key(tar_key: str) -> dict[str, str]:
    """Parse a tar key like 'target/atomic/OpenDrawer/20250816/lerobot.tar'."""
    parts = tar_key.replace(".tar", "").split("/")
    if len(parts) == 5:
        return {
            "split": parts[0],
            "category": parts[1],
            "task": parts[2],
            "date": parts[3],
            "kind": parts[4],
        }
    # Pretrain mimicgen paths are deeper, treat source as mimicgen.
    if len(parts) > 5 and parts[0] == "pretrain":
        return {
            "split": parts[0],
            "category": parts[1],
            "task": parts[2],
            "date": parts[3],
            "kind": parts[-1],
            "source": "mimicgen",
        }
    return {"split": "unknown", "category": "unknown", "task": tar_key, "date": "", "kind": "lerobot"}


def _resolve_datasets(
    box_links: dict[str, str],
    *,
    subset: str,
    tasks: list[str] | None,
    splits: list[str] | None,
    sources: list[str] | None,
) -> list[tuple[str, str, str]]:
    """Resolve the list of tar keys to download.

    Returns [(tar_key, label, source)], where source is 'human' or 'mimicgen'.
    """
    candidates: list[tuple[str, str, str]] = []

    if subset == "current":
        for task_name, date in CURRENT_CONFIG_TASKS:
            if tasks and task_name not in tasks:
                continue
            tar_key = f"target/composite/{task_name}/{date}/lerobot.tar"
            if tar_key in box_links:
                candidates.append((tar_key, f"current/{task_name}", "human"))
    else:
        for tar_key in sorted(box_links):
            info = _parse_tar_key(tar_key)
            src = info.get("source", "human")
            sp = info["split"]
            cat = info["category"]
            task = info["task"]

            # Subset filter
            if subset == "all":
                pass
            elif subset in ("target", "pretrain"):
                if sp != subset:
                    continue
            elif "/" in subset:
                want_sp, want_cat = subset.split("/", 1)
                if sp != want_sp or cat != want_cat:
                    continue
            else:
                continue

            # Task name filter
            if tasks and task not in tasks:
                continue

            # Split filter
            if splits and sp not in splits:
                continue

            # Source filter
            if sources and src not in sources:
                continue

            candidates.append((tar_key, f"{sp}/{cat}/{task}", src))

    return candidates


def _format_bytes(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}P"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download RoboCasa365 datasets with resume support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subset",
        default=None,
        choices=SUBSET_CHOICES,
        help="Dataset subset to download. Defaults to 'current' unless --tasks is used, in which case it searches all.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Filter to specific task names (e.g., OpenDrawer SeparateFreezerRack).",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["target", "pretrain"],
        default=None,
        help="Filter by split.",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        choices=["human", "mimicgen"],
        default=None,
        help="Filter by source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without downloading. This is also the default.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform network downloads and extraction. Default only prints the plan.",
    )
    parser.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep downloaded tar archives after extraction.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Per-file download timeout in seconds. Default: 7200 (2 hours).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and overwrite already-extracted datasets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    box_links = _load_box_links()
    base_path = _dataset_base_path()

    # Default to current config; if --tasks is provided without --subset, search all datasets.
    subset = args.subset if args.subset is not None else ("all" if args.tasks else "current")

    datasets = _resolve_datasets(
        box_links,
        subset=subset,
        tasks=args.tasks,
        splits=args.split,
        sources=args.source,
    )

    if not datasets:
        print("No datasets matched the requested subset/filters.")
        return 0

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for tar_key, label, src in datasets:
        if tar_key not in seen:
            seen.add(tar_key)
            unique.append((tar_key, label, src))
    datasets = unique

    # Categorize by existing state.
    to_download: list[tuple[str, str, str, Path, Path]] = []
    already_done: list[tuple[str, str]] = []
    for tar_key, label, src in datasets:
        ds_path = base_path / tar_key.replace(".tar", "")
        extract_dir = ds_path.parent
        tar_path = extract_dir / "lerobot.tar"
        if _is_extracted(ds_path) and not args.overwrite:
            already_done.append((label, str(ds_path)))
        else:
            to_download.append((tar_key, label, src, ds_path, tar_path))

    print(f"Subset: {subset}")
    print(f"Total matched: {len(datasets)}")
    print(f"Already extracted: {len(already_done)}")
    print(f"To download: {len(to_download)}")

    if args.dry_run or not args.execute:
        print("\n=== Dry run plan ===")
        for tar_key, label, src, ds_path, tar_path in to_download:
            partial = tar_path.stat().st_size if tar_path.exists() else 0
            partial_str = f" (partial {_format_bytes(partial)})" if partial else ""
            print(f"  {label} [{src}] -> {tar_path}{partial_str}")
        for label, ds_str in already_done:
            print(f"  {label} -> already present at {ds_str}")
        print("\nPass --execute to download and extract missing datasets.")
        return 0

    failed: list[str] = []
    succeeded: list[str] = []
    skipped: list[str] = [label for label, _ in already_done]

    for index, (tar_key, label, src, ds_path, tar_path) in enumerate(to_download, start=1):
        print(f"\n[{index}/{len(to_download)}] === {label} ({src}) ===")
        print(f"[info] tar: {tar_path}")
        print(f"[info] dataset: {ds_path}")

        shared_url = box_links[tar_key]
        url = _get_direct_download_url(shared_url, ext="tar")

        # Remove stale partial if overwrite requested.
        if args.overwrite and tar_path.exists():
            print("[overwrite] removing existing partial tar")
            tar_path.unlink()

        # Download (resume if partial).
        if not tar_path.exists() or tar_path.stat().st_size == 0:
            ok = _download_with_wget(url, tar_path, timeout=args.timeout)
            if not ok:
                ok = _download_with_curl(url, tar_path, timeout=args.timeout)
        else:
            print(f"[resume] partial tar {_format_bytes(tar_path.stat().st_size)}, resuming")
            ok = _download_with_wget(url, tar_path, timeout=args.timeout)
            if not ok:
                ok = _download_with_curl(url, tar_path, timeout=args.timeout)

        if not ok:
            print(f"[fail] download failed for {label}")
            failed.append(label)
            continue

        # Extract.
        if _extract_and_cleanup(tar_path, extract_dir, keep_tar=args.keep_tar):
            succeeded.append(label)
        else:
            failed.append(label)
            if not args.keep_tar:
                print("[cleanup] keeping tar for inspection due to extraction failure")

    print("\n=== Summary ===")
    print(f"Succeeded: {len(succeeded)}/{len(to_download)}")
    for label in succeeded:
        print(f"  ✓ {label}")
    if skipped:
        print(f"Skipped (already present): {len(skipped)}")
        for label in skipped:
            print(f"  ○ {label}")
    if failed:
        print(f"Failed: {len(failed)}/{len(to_download)}")
        for label in failed:
            print(f"  ✗ {label}")
        return 1

    print("All requested datasets are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
