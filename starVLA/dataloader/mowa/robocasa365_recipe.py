"""MoWA RoboCasa365 fixed data recipes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starVLA.dataloader.mowa.schema import DATA_GATE


MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_RECIPE = (
    "mowa_robocasa365_target_human_atomic_core_v1"
)

MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS = {
    "OpenDrawer": "v1.0/target/atomic/OpenDrawer/20250816/lerobot",
    "OpenCabinet": "v1.0/target/atomic/OpenCabinet/20250813/lerobot",
    "CloseFridge": "v1.0/target/atomic/CloseFridge/20250816/lerobot",
    "CloseToasterOvenDoor": "v1.0/target/atomic/CloseToasterOvenDoor/20250818/lerobot",
    "CoffeeSetupMug": "v1.0/target/atomic/CoffeeSetupMug/20250813/lerobot",
    "NavigateKitchen": "v1.0/target/atomic/NavigateKitchen/20250821/lerobot",
    "PickPlaceCounterToCabinet": "v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot",
    "PickPlaceToasterToCounter": "v1.0/target/atomic/PickPlaceToasterToCounter/20250817/lerobot",
    "PickPlaceSinkToCounter": "v1.0/target/atomic/PickPlaceSinkToCounter/20250813/lerobot",
    "TurnOnSinkFaucet": "v1.0/target/atomic/TurnOnSinkFaucet/20250812/lerobot",
}


@dataclass(frozen=True)
class MoWARoboCasa365RecipeTaskStatus:
    task: str
    relative_path: str
    expected_path: str
    path_exists: bool
    meta_exists: bool
    data_exists: bool
    videos_exists: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "relative_path": self.relative_path,
            "expected_path": self.expected_path,
            "path_exists": self.path_exists,
            "meta_exists": self.meta_exists,
            "data_exists": self.data_exists,
            "videos_exists": self.videos_exists,
        }


@dataclass(frozen=True)
class MoWARoboCasa365RecipeStatus:
    recipe_name: str
    split: str
    source: str
    task_count: int
    available_task_count: int
    missing_task_count: int
    tasks: tuple[MoWARoboCasa365RecipeTaskStatus, ...]
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "split": self.split,
            "source": self.source,
            "task_count": self.task_count,
            "available_task_count": self.available_task_count,
            "missing_task_count": self.missing_task_count,
            "tasks": [task.to_dict() for task in self.tasks],
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


def inspect_mowa_robocasa365_atomic_core_recipe(
    data_root: Path | str,
) -> MoWARoboCasa365RecipeStatus:
    """检查 MoWA target/human atomic core recipe 是否已下载齐全。"""

    root = Path(data_root)
    tasks = []
    for task, relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.items():
        expected_path = root / relative_path
        tasks.append(
            MoWARoboCasa365RecipeTaskStatus(
                task=task,
                relative_path=relative_path,
                expected_path=str(expected_path),
                path_exists=expected_path.is_dir(),
                meta_exists=(expected_path / "meta" / "episodes.jsonl").is_file(),
                data_exists=(expected_path / "data").is_dir(),
                videos_exists=(expected_path / "videos").is_dir(),
            )
        )

    available = tuple(
        task
        for task in tasks
        if task.path_exists and task.meta_exists and task.data_exists and task.videos_exists
    )
    missing_count = len(tasks) - len(available)
    return MoWARoboCasa365RecipeStatus(
        recipe_name=MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_RECIPE,
        split="target",
        source="human",
        task_count=len(tasks),
        available_task_count=len(available),
        missing_task_count=missing_count,
        tasks=tuple(tasks),
        go_no_go=(
            "No-Go: recipe data incomplete"
            if missing_count
            else "TBD: recipe available; profile/leakage/labels still Data Gate"
        ),
        notes=(
            "This recipe is the fixed MoWA primary comparison data domain.",
            "Availability does not imply G0 pass; obs fps/action Hz/window/labels remain Data Gate.",
            f"All task paths are registry-confirmed target/human atomic tasks; runtime profile status is {DATA_GATE}.",
        ),
    )
