"""MoWA future label source audit utilities."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from starVLA.dataloader.mowa.robocasa365_recipe import (
    MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_RECIPE,
MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS,
)
from starVLA.dataloader.mowa.schema import DATA_GATE


OPENDRAWER_SUBGOAL_SCHEMA_VERSION = "opendrawer_open_drawer_v1"
OPENDRAWER_SUBGOAL_ID = "open_drawer"
OPENDRAWER_SUBGOAL_COMPLETION_THRESHOLD = 0.95
OPENDRAWER_READINESS_SCHEMA_VERSION = "opendrawer_open_proxy_v1"
OPENDRAWER_READINESS_TYPE = "open"
OPENDRAWER_READINESS_TARGET = "drawer"


PANDA_OMRON_MODALITY_PATH = Path(
    "playground/Code/robocasa365/robocasa/models/assets/groot_dataset_assets/"
    "PandaOmron_modality.json"
)


@dataclass(frozen=True)
class MoWAFutureLabelTaskAudit:
    task: str
    dataset_path: str
    path_exists: bool
    sampled_episode_indices: tuple[int, ...]
    parquet_columns: tuple[str, ...]
    state_schema: dict[str, Any]
    reward_done_audit: dict[str, Any]
    extras_audit: dict[str, Any]
    task_source_audit: dict[str, Any]
    label_readiness: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "dataset_path": self.dataset_path,
            "path_exists": self.path_exists,
            "sampled_episode_indices": self.sampled_episode_indices,
            "parquet_columns": self.parquet_columns,
            "state_schema": self.state_schema,
            "reward_done_audit": self.reward_done_audit,
            "extras_audit": self.extras_audit,
            "task_source_audit": self.task_source_audit,
            "label_readiness": self.label_readiness,
        }


@dataclass(frozen=True)
class MoWAFutureLabelSourceAuditReport:
    recipe_name: str
    data_root: str
    task_count: int
    available_task_count: int
    sampled_episode_indices: tuple[int, ...]
    tasks: tuple[MoWAFutureLabelTaskAudit, ...]
    summary: dict[str, Any]
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "data_root": self.data_root,
            "task_count": self.task_count,
            "available_task_count": self.available_task_count,
            "sampled_episode_indices": self.sampled_episode_indices,
            "tasks": [task.to_dict() for task in self.tasks],
            "summary": self.summary,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MoWAFailureRiskTaskSmoke:
    task: str
    dataset_path: str
    path_exists: bool
    episode_count: int
    row_count: int
    labeled_count: int
    positive_count: int
    negative_count: int
    coverage: float
    positive_rate: float | None
    next_done_positive_rate: float | None
    next_done_phi_correlation: float | None
    reward_unique_values_preview: tuple[float, ...]
    reward_is_sparse_0_1: bool
    gate_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "dataset_path": self.dataset_path,
            "path_exists": self.path_exists,
            "episode_count": self.episode_count,
            "row_count": self.row_count,
            "labeled_count": self.labeled_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "coverage": self.coverage,
            "positive_rate": self.positive_rate,
            "next_done_positive_rate": self.next_done_positive_rate,
            "next_done_phi_correlation": self.next_done_phi_correlation,
            "reward_unique_values_preview": self.reward_unique_values_preview,
            "reward_is_sparse_0_1": self.reward_is_sparse_0_1,
            "gate_status": self.gate_status,
        }


@dataclass(frozen=True)
class MoWAFailureRiskDataGateSmoke:
    recipe_name: str
    data_root: str
    horizon: int
    task_count: int
    available_task_count: int
    tasks: tuple[MoWAFailureRiskTaskSmoke, ...]
    summary: dict[str, Any]
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "data_root": self.data_root,
            "horizon": self.horizon,
            "task_count": self.task_count,
            "available_task_count": self.available_task_count,
            "tasks": [task.to_dict() for task in self.tasks],
            "summary": self.summary,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MoWAOpenDrawerEpisodeMappingSample:
    episode_index: int
    state_length: int
    drawer_qpos_first: float
    drawer_qpos_last: float
    drawer_qpos_min: float
    drawer_qpos_max: float
    drawer_normalized_open_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "state_length": self.state_length,
            "drawer_qpos_first": self.drawer_qpos_first,
            "drawer_qpos_last": self.drawer_qpos_last,
            "drawer_qpos_min": self.drawer_qpos_min,
            "drawer_qpos_max": self.drawer_qpos_max,
            "drawer_normalized_open_max": self.drawer_normalized_open_max,
        }


@dataclass(frozen=True)
class MoWAOpenDrawerStateMappingAudit:
    dataset_path: str
    fixture_ref_name: str
    fixture_ref_value: str
    drawer_joint_name: str
    drawer_joint_range: tuple[float, float]
    state_vector_width: int
    inferred_nq: int
    inferred_nv: int
    drawer_qpos_state_index: int
    drawer_qvel_state_index: int
    sample_episode_count: int
    sampled_episodes: tuple[MoWAOpenDrawerEpisodeMappingSample, ...]
    summary: dict[str, Any]
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "fixture_ref_name": self.fixture_ref_name,
            "fixture_ref_value": self.fixture_ref_value,
            "drawer_joint_name": self.drawer_joint_name,
            "drawer_joint_range": self.drawer_joint_range,
            "state_vector_width": self.state_vector_width,
            "inferred_nq": self.inferred_nq,
            "inferred_nv": self.inferred_nv,
            "drawer_qpos_state_index": self.drawer_qpos_state_index,
            "drawer_qvel_state_index": self.drawer_qvel_state_index,
            "sample_episode_count": self.sample_episode_count,
            "sampled_episodes": [sample.to_dict() for sample in self.sampled_episodes],
            "summary": self.summary,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MoWAOpenDrawerSubgoalEpisodeSmoke:
    episode_index: int
    row_count: int
    state_length: int
    terminal_reward: float
    terminal_done: bool
    terminal_success_proxy: bool
    drawer_progress_max: float
    drawer_progress_terminal: float
    state_success_proxy: bool
    anchor_positive_count: int
    anchor_negative_count: int
    anchor_positive_rate: float
    alignment_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "row_count": self.row_count,
            "state_length": self.state_length,
            "terminal_reward": self.terminal_reward,
            "terminal_done": self.terminal_done,
            "terminal_success_proxy": self.terminal_success_proxy,
            "drawer_progress_max": self.drawer_progress_max,
            "drawer_progress_terminal": self.drawer_progress_terminal,
            "state_success_proxy": self.state_success_proxy,
            "anchor_positive_count": self.anchor_positive_count,
            "anchor_negative_count": self.anchor_negative_count,
            "anchor_positive_rate": self.anchor_positive_rate,
            "alignment_status": self.alignment_status,
        }


@dataclass(frozen=True)
class MoWAOpenDrawerSubgoalDataGateSmoke:
    dataset_path: str
    horizon: int
    episode_count: int
    sampled_episodes: tuple[MoWAOpenDrawerSubgoalEpisodeSmoke, ...]
    summary: dict[str, Any]
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "horizon": self.horizon,
            "episode_count": self.episode_count,
            "sampled_episodes": [episode.to_dict() for episode in self.sampled_episodes],
            "summary": self.summary,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MoWAOpenDrawerStepLabel:
    timestep: int
    frame_index: int
    drawer_progress: float
    drawer_qvel: float
    predicates: dict[str, Any]
    labels: dict[str, Any]
    masks: dict[str, bool]
    debug: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "frame_index": self.frame_index,
            "drawer_progress": self.drawer_progress,
            "drawer_qvel": self.drawer_qvel,
            "predicates": self.predicates,
            "labels": self.labels,
            "masks": self.masks,
            "debug": self.debug,
        }


@dataclass(frozen=True)
class MoWAOpenDrawerFutureLabelEpisodeSidecar:
    episode_index: int
    row_count: int
    state_length: int
    schema_contract: dict[str, Any]
    terminal_success_proxy: bool
    state_success_proxy: bool
    alignment_status: str
    steps: tuple[MoWAOpenDrawerStepLabel, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "row_count": self.row_count,
            "state_length": self.state_length,
            "schema_contract": self.schema_contract,
            "terminal_success_proxy": self.terminal_success_proxy,
            "state_success_proxy": self.state_success_proxy,
            "alignment_status": self.alignment_status,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class MoWAOpenDrawerFutureLabelSidecarSmoke:
    dataset_path: str
    failure_risk_horizon: int
    subgoal_horizon: int
    readiness_horizon: int
    readiness_progress_delta: float
    schema_contract: dict[str, Any]
    episode_count: int
    sampled_episodes: tuple[MoWAOpenDrawerFutureLabelEpisodeSidecar, ...]
    summary: dict[str, Any]
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "failure_risk_horizon": self.failure_risk_horizon,
            "subgoal_horizon": self.subgoal_horizon,
            "readiness_horizon": self.readiness_horizon,
            "readiness_progress_delta": self.readiness_progress_delta,
            "schema_contract": self.schema_contract,
            "episode_count": self.episode_count,
            "sampled_episodes": [episode.to_dict() for episode in self.sampled_episodes],
            "summary": self.summary,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


def build_mowa_future_label_source_audit(
    data_root: Path | str,
    repo_root: Path | str = ".",
    episode_indices: tuple[int, ...] = (0, 1, 4),
) -> MoWAFutureLabelSourceAuditReport:
    """Audit source fields for non-vision future label proxies.

    This is a read-only G0 audit. It does not generate training labels, decode videos,
    replay MuJoCo, or modify any production dataloader masks.
    """

    root = Path(data_root)
    repo = Path(repo_root)
    tasks = tuple(
        _audit_task(
            task=task,
            dataset_path=root / relative_path,
            repo_root=repo,
            episode_indices=episode_indices,
        )
        for task, relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.items()
    )
    available = tuple(task for task in tasks if task.path_exists)
    failure_risk_ready = sum(
        task.label_readiness["failure_risk"] == "candidate_for_data_gate" for task in tasks
    )
    subgoal_blocked = sum(task.label_readiness["subgoal_feasibility"].startswith("blocked") for task in tasks)
    readiness_blocked = sum(task.label_readiness["manipulation_readiness"].startswith("blocked") for task in tasks)

    return MoWAFutureLabelSourceAuditReport(
        recipe_name=MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_RECIPE,
        data_root=str(root),
        task_count=len(tasks),
        available_task_count=len(available),
        sampled_episode_indices=episode_indices,
        tasks=tasks,
        summary={
            "failure_risk_candidate_task_count": failure_risk_ready,
            "subgoal_feasibility_blocked_task_count": subgoal_blocked,
            "manipulation_readiness_blocked_task_count": readiness_blocked,
            "state_schema_source": "PandaOmron_modality.json or dataset meta/modality.json",
            "extras_required_for_subgoal_or_readiness": True,
        },
        go_no_go=(
            "TBD: source audit complete; label builders remain gated"
            if len(available) == len(tasks)
            else "No-Go: source audit incomplete because recipe data is missing"
        ),
        notes=(
            "failure_risk can only move to data-gate if reward/done distribution is usable.",
            "subgoal_feasibility requires task schema plus simulator/object predicate fields.",
            "manipulation_readiness requires state semantics plus readiness/onset predicates.",
            f"No production masks are changed by this {DATA_GATE} audit.",
        ),
    )


def build_mowa_opendrawer_future_label_sidecar_smoke(
    dataset_path: Path | str,
    failure_risk_horizon: int = 10,
    subgoal_horizon: int = 20,
    readiness_horizon: int = 5,
    readiness_progress_delta: float = 0.10,
    max_episodes: int | None = 2,
) -> MoWAOpenDrawerFutureLabelSidecarSmoke:
    """Build a read-only OpenDrawer future-label sidecar smoke report.

    This is an audit-side sidecar generator, not a production dataloader builder.
    It only uses existing reward/done and simulator-state sources to emit weak labels
    for inspection. No production masks or training paths are modified.
    """

    root = Path(dataset_path)
    parquet_paths = sorted((root / "data" / "chunk-000").glob("episode_*.parquet"))
    if max_episodes is not None:
        parquet_paths = parquet_paths[:max_episodes]

    schema_contract = _build_opendrawer_schema_contract(readiness_progress_delta)
    episodes = tuple(
        _build_opendrawer_future_label_episode_sidecar(
            parquet_path=parquet_path,
            states_path=root / "extras" / parquet_path.stem / "states.npz",
            failure_risk_horizon=failure_risk_horizon,
            subgoal_horizon=subgoal_horizon,
            readiness_horizon=readiness_horizon,
            readiness_progress_delta=readiness_progress_delta,
            schema_contract=schema_contract,
        )
        for parquet_path in parquet_paths
        if (root / "extras" / parquet_path.stem / "states.npz").is_file()
    )

    all_steps = [step for episode in episodes for step in episode.steps]
    failure_risk_labeled = sum(step.masks["failure_risk"] for step in all_steps)
    failure_risk_positive = sum(
        step.masks["failure_risk"] and float(step.labels["failure_risk"]) > 0.0 for step in all_steps
    )
    subgoal_labeled = sum(step.masks["subgoal_feasibility"] for step in all_steps)
    subgoal_positive = sum(
        step.masks["subgoal_feasibility"] and float(step.labels["subgoal_feasibility"]) > 0.0
        for step in all_steps
    )
    readiness_labeled = sum(step.masks["manipulation_readiness"] for step in all_steps)
    readiness_positive = sum(
        step.masks["manipulation_readiness"] and float(step.labels["manipulation_readiness"]) > 0.0
        for step in all_steps
    )
    mismatch = sum(episode.alignment_status != "aligned_success" for episode in episodes)
    total_steps = len(all_steps)

    if mismatch > 0:
        go_no_go = "No-Go: OpenDrawer sidecar remains audit-only because state-vs-terminal mismatch persists"
    else:
        go_no_go = "TBD: OpenDrawer sidecar smoke built; labels remain audit-only until mask-lift review"

    return MoWAOpenDrawerFutureLabelSidecarSmoke(
        dataset_path=str(root),
        failure_risk_horizon=failure_risk_horizon,
        subgoal_horizon=subgoal_horizon,
        readiness_horizon=readiness_horizon,
        readiness_progress_delta=readiness_progress_delta,
        schema_contract=schema_contract,
        episode_count=len(episodes),
        sampled_episodes=episodes,
        summary={
            "step_count": total_steps,
            "failure_risk_labeled_count": failure_risk_labeled,
            "failure_risk_positive_count": failure_risk_positive,
            "subgoal_labeled_count": subgoal_labeled,
            "subgoal_positive_count": subgoal_positive,
            "manipulation_readiness_labeled_count": readiness_labeled,
            "manipulation_readiness_positive_count": readiness_positive,
            "alignment_mismatch_episode_count": mismatch,
        },
        go_no_go=go_no_go,
        notes=(
            "failure_risk reuses the existing done/reward H-step proxy.",
            "subgoal_feasibility fixes the next subgoal to OpenDrawer/open_drawer and uses drawer-progress predicate.",
            "manipulation_readiness is a weak proxy: future K-step drawer-progress gain >= readiness_progress_delta.",
            f"No production masks are changed by this {DATA_GATE} sidecar smoke.",
        ),
    )


def build_mowa_opendrawer_subgoal_data_gate_smoke(
    dataset_path: Path | str,
    horizon: int = 20,
    max_episodes: int | None = None,
) -> MoWAOpenDrawerSubgoalDataGateSmoke:
    """Quantify single-subgoal feasibility for OpenDrawer/open_drawer."""

    root = Path(dataset_path)
    parquet_paths = sorted((root / "data" / "chunk-000").glob("episode_*.parquet"))
    if max_episodes is not None:
        parquet_paths = parquet_paths[:max_episodes]

    episodes = tuple(
        _build_opendrawer_subgoal_episode_smoke(
            parquet_path=parquet_path,
            states_path=root / "extras" / parquet_path.stem / "states.npz",
            horizon=horizon,
        )
        for parquet_path in parquet_paths
        if (root / "extras" / parquet_path.stem / "states.npz").is_file()
    )

    positive = sum(episode.anchor_positive_count for episode in episodes)
    negative = sum(episode.anchor_negative_count for episode in episodes)
    total = positive + negative
    state_success = sum(episode.state_success_proxy for episode in episodes)
    terminal_success = sum(episode.terminal_success_proxy for episode in episodes)
    aligned = sum(episode.alignment_status == "aligned_success" for episode in episodes)
    mismatch = sum(episode.alignment_status != "aligned_success" for episode in episodes)

    if positive == 0 or negative == 0:
        go_no_go = "No-Go: OpenDrawer open_drawer single-subgoal labels are single-class"
    elif mismatch > 0:
        go_no_go = "No-Go: OpenDrawer open_drawer subgoal labels show state-vs-terminal alignment mismatch"
    else:
        go_no_go = "TBD: OpenDrawer open_drawer subgoal labels look usable; builder remains gated"

    return MoWAOpenDrawerSubgoalDataGateSmoke(
        dataset_path=str(root),
        horizon=horizon,
        episode_count=len(episodes),
        sampled_episodes=episodes[:32],
        summary={
            "anchor_positive_count": positive,
            "anchor_negative_count": negative,
            "anchor_positive_rate": (positive / total) if total else None,
            "state_success_episode_count": state_success,
            "terminal_success_episode_count": terminal_success,
            "aligned_success_episode_count": aligned,
            "mismatched_episode_count": mismatch,
        },
        go_no_go=go_no_go,
        notes=(
            "Anchor labels use the single OpenDrawer subgoal: window reaches normalized drawer progress >= 0.95.",
            "state_success_proxy is based on simulator-state drawer progress, not reward/done.",
            "terminal_success_proxy is next.done && next.reward > 0 on the last parquet row.",
            "Episode-specific drawer joint mapping is derived from each episode's ep_meta/model.xml.gz.",
            "Any mismatch means builder integration remains gated until episode alignment is understood.",
        ),
    )


def build_mowa_opendrawer_state_mapping_audit(
    dataset_path: Path | str,
    episode_indices: tuple[int, ...] = (0, 1, 4, 7),
) -> MoWAOpenDrawerStateMappingAudit:
    """Audit OpenDrawer simulator-state mapping for drawer progress predicates."""

    root = Path(dataset_path)
    ep_meta = json.loads((root / "extras" / "episode_000000" / "ep_meta.json").read_text(encoding="utf-8"))
    fixture_ref_value = ep_meta["fixture_refs"]["drawer"]
    xml_root = ET.fromstring(gzip.decompress((root / "extras" / "episode_000000" / "model.xml.gz").read_bytes()))
    ordered_joints = _ordered_joint_state_layout(xml_root)
    drawer_joint_name = f"{fixture_ref_value}_slidejoint"
    drawer_entry = next(entry for entry in ordered_joints if entry["name"] == drawer_joint_name)
    state_width = int(np.load(root / "extras" / "episode_000000" / "states.npz", allow_pickle=True)["states"].shape[1])
    inferred_nq = sum(int(entry["nq"]) for entry in ordered_joints)
    inferred_nv = sum(int(entry["nv"]) for entry in ordered_joints)
    drawer_qpos_index = int(drawer_entry["qpos_index"])
    drawer_qvel_index = 1 + inferred_nq + int(drawer_entry["qvel_index"]) - 1
    joint_range = tuple(float(value) for value in drawer_entry["range"])
    open_denominator = abs(joint_range[0]) * 0.55 / 2

    samples = tuple(
        _build_opendrawer_episode_mapping_sample(
            states_path=root / "extras" / f"episode_{episode_index:06d}" / "states.npz",
            episode_index=episode_index,
            drawer_qpos_index=drawer_qpos_index,
            open_denominator=open_denominator,
        )
        for episode_index in episode_indices
        if (root / "extras" / f"episode_{episode_index:06d}" / "states.npz").is_file()
    )
    sampled_qpos_indices = tuple(
        _resolve_opendrawer_drawer_progress_layout(root / "extras" / f"episode_{episode_index:06d}" / "states.npz")[0]
        for episode_index in episode_indices
        if (root / "extras" / f"episode_{episode_index:06d}" / "states.npz").is_file()
    )
    opened_samples = sum(sample.drawer_normalized_open_max >= 0.95 for sample in samples)

    return MoWAOpenDrawerStateMappingAudit(
        dataset_path=str(root),
        fixture_ref_name="drawer",
        fixture_ref_value=fixture_ref_value,
        drawer_joint_name=drawer_joint_name,
        drawer_joint_range=joint_range,
        state_vector_width=state_width,
        inferred_nq=inferred_nq,
        inferred_nv=inferred_nv,
        drawer_qpos_state_index=drawer_qpos_index,
        drawer_qvel_state_index=drawer_qvel_index,
        sample_episode_count=len(samples),
        sampled_episodes=samples,
        summary={
            "state_layout_matches_time_plus_qpos_plus_qvel": state_width == 1 + inferred_nq + inferred_nv,
            "open_denominator_from_joint_range": open_denominator,
            "episodes_reaching_success_threshold_on_sample": opened_samples,
            "sampled_episode_indices": episode_indices,
            "sampled_drawer_qpos_state_indices": sampled_qpos_indices,
            "episode_specific_joint_index_variation_observed": len(set(sampled_qpos_indices)) > 1,
            "subgoal_open_drawer_predicate_ready": True,
            "manipulation_readiness_predicate_ready": False,
        },
        go_no_go=(
            "TBD: OpenDrawer drawer-progress predicate is source-ready for subgoal schema drafting"
            if state_width == 1 + inferred_nq + inferred_nv
            else "No-Go: OpenDrawer state mapping audit failed to match flattened sim-state layout"
        ),
        notes=(
            "states.npz stores flattened simulator state copied from HDF5 demo states.",
            "Playback code compares env.sim.get_state().flatten() against stored states[t+1].",
            "For drawers, raw qpos is negative when opened; normalized progress uses (-qpos) / (0.55 * abs(range_min) / 2).",
            "The drawer qpos index is episode-specific across OpenDrawer layouts and must be resolved from each episode's ep_meta/model.xml.gz.",
            "This audit only establishes drawer-progress mapping for OpenDrawer; it does not unblock manipulation_readiness.",
        ),
    )


def build_mowa_failure_risk_data_gate_smoke(
    data_root: Path | str,
    horizon: int = 10,
    max_episodes_per_task: int | None = None,
) -> MoWAFailureRiskDataGateSmoke:
    """Quantify failure_risk proxy coverage and distribution on atomic-core tasks."""

    root = Path(data_root)
    tasks = tuple(
        _build_failure_risk_task_smoke(
            task=task,
            dataset_path=root / relative_path,
            horizon=horizon,
            max_episodes=max_episodes_per_task,
        )
        for task, relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.items()
    )
    available = tuple(task for task in tasks if task.path_exists)
    sparse_reward_pass = sum(task.reward_is_sparse_0_1 for task in tasks)
    both_classes = sum(task.positive_count > 0 and task.negative_count > 0 for task in tasks)
    low_coverage = sum(task.coverage < 0.05 for task in tasks if task.row_count > 0)
    high_correlation = sum(
        task.next_done_phi_correlation is not None and abs(task.next_done_phi_correlation) >= 0.8
        for task in tasks
    )
    candidate_tasks = sum(task.gate_status == "candidate_for_mask_lift_review" for task in tasks)

    if len(available) != len(tasks):
        go_no_go = "No-Go: failure_risk data gate incomplete because recipe data is missing"
    elif candidate_tasks == 0:
        go_no_go = "No-Go: failure_risk proxy remains gated by single-class or unusable distribution"
    else:
        go_no_go = "TBD: failure_risk proxy is constructible; review low-coverage and correlation risk before unmasking"

    return MoWAFailureRiskDataGateSmoke(
        recipe_name=MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_RECIPE,
        data_root=str(root),
        horizon=horizon,
        task_count=len(tasks),
        available_task_count=len(available),
        tasks=tasks,
        summary={
            "sparse_reward_task_count": sparse_reward_pass,
            "both_classes_task_count": both_classes,
            "low_coverage_task_count": low_coverage,
            "high_next_done_correlation_task_count": high_correlation,
            "candidate_for_mask_lift_review_task_count": candidate_tasks,
        },
        go_no_go=go_no_go,
        notes=(
            "failure_risk uses the H-step done/reward proxy from 09_future_label_builder_design.md.",
            "Coverage below 5% is not an automatic block, but must be recorded as training-value risk.",
            "next_done correlation uses phi coefficient on labeled anchors only.",
            f"No production masks are changed by this {DATA_GATE} smoke.",
        ),
    )


def _audit_task(
    task: str,
    dataset_path: Path,
    repo_root: Path,
    episode_indices: tuple[int, ...],
) -> MoWAFutureLabelTaskAudit:
    path_exists = dataset_path.is_dir()
    columns = _read_union_parquet_columns(dataset_path, episode_indices)
    state_schema = _read_state_schema(dataset_path, repo_root)
    reward_done_audit = _audit_reward_done(dataset_path, episode_indices)
    extras_audit = _audit_extras(dataset_path, episode_indices)
    task_source_audit = _audit_task_source(task, repo_root)
    label_readiness = _build_label_readiness(
        columns=columns,
        reward_done_audit=reward_done_audit,
        extras_audit=extras_audit,
        state_schema=state_schema,
    )

    return MoWAFutureLabelTaskAudit(
        task=task,
        dataset_path=str(dataset_path),
        path_exists=path_exists,
        sampled_episode_indices=episode_indices,
        parquet_columns=columns,
        state_schema=state_schema,
        reward_done_audit=reward_done_audit,
        extras_audit=extras_audit,
        task_source_audit=task_source_audit,
        label_readiness=label_readiness,
    )


def _build_failure_risk_task_smoke(
    task: str,
    dataset_path: Path,
    horizon: int,
    max_episodes: int | None,
) -> MoWAFailureRiskTaskSmoke:
    if not dataset_path.is_dir():
        return MoWAFailureRiskTaskSmoke(
            task=task,
            dataset_path=str(dataset_path),
            path_exists=False,
            episode_count=0,
            row_count=0,
            labeled_count=0,
            positive_count=0,
            negative_count=0,
            coverage=0.0,
            positive_rate=None,
            next_done_positive_rate=None,
            next_done_phi_correlation=None,
            reward_unique_values_preview=(),
            reward_is_sparse_0_1=False,
            gate_status="blocked_by_missing_dataset_path",
        )

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA failure_risk data gate smoke requires pyarrow.") from exc

    parquet_paths = sorted((dataset_path / "data" / "chunk-000").glob("episode_*.parquet"))
    if max_episodes is not None:
        parquet_paths = parquet_paths[:max_episodes]

    row_count = 0
    labeled_count = 0
    positive_count = 0
    negative_count = 0
    reward_values: set[float] = set()
    labeled_pairs: list[tuple[int, int]] = []

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=["next.reward", "next.done"])
        data = table.to_pydict()
        rewards = [float(value) for value in data.get("next.reward", ())]
        dones = [bool(value) for value in data.get("next.done", ())]
        row_count += len(rewards)
        reward_values.update(rewards)

        for anchor, anchor_done in enumerate(dones):
            label = _failure_risk_label_from_window(rewards, dones, anchor, horizon=horizon)
            if label is None:
                continue
            labeled_count += 1
            if label > 0:
                positive_count += 1
            else:
                negative_count += 1
            labeled_pairs.append((1 if anchor_done else 0, 1 if label > 0 else 0))

    coverage = (labeled_count / row_count) if row_count else 0.0
    positive_rate = (positive_count / labeled_count) if labeled_count else None
    next_done_positive_rate = (
        sum(anchor_done for anchor_done, _ in labeled_pairs) / labeled_count if labeled_count else None
    )
    phi_correlation = _binary_phi_coefficient(labeled_pairs)
    reward_is_sparse_0_1 = reward_values.issubset({0.0, 1.0}) if reward_values else False

    if not reward_is_sparse_0_1:
        gate_status = "blocked_by_non_sparse_reward_distribution"
    elif labeled_count == 0:
        gate_status = "blocked_by_zero_labeled_anchors"
    elif positive_count == 0 or negative_count == 0:
        gate_status = "blocked_by_single_class_distribution"
    else:
        gate_status = "candidate_for_mask_lift_review"

    return MoWAFailureRiskTaskSmoke(
        task=task,
        dataset_path=str(dataset_path),
        path_exists=True,
        episode_count=len(parquet_paths),
        row_count=row_count,
        labeled_count=labeled_count,
        positive_count=positive_count,
        negative_count=negative_count,
        coverage=coverage,
        positive_rate=positive_rate,
        next_done_positive_rate=next_done_positive_rate,
        next_done_phi_correlation=phi_correlation,
        reward_unique_values_preview=tuple(sorted(reward_values))[:16],
        reward_is_sparse_0_1=reward_is_sparse_0_1,
        gate_status=gate_status,
    )


def _read_union_parquet_columns(dataset_path: Path, episode_indices: tuple[int, ...]) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA future label source audit requires pyarrow.") from exc

    columns: set[str] = set()
    for episode_index in episode_indices:
        parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        if parquet_path.is_file():
            columns.update(pq.ParquetFile(parquet_path).schema_arrow.names)
    return tuple(sorted(columns))


def _read_state_schema(dataset_path: Path, repo_root: Path) -> dict[str, Any]:
    modality_paths = (
        dataset_path / "meta" / "modality.json",
        repo_root / PANDA_OMRON_MODALITY_PATH,
    )
    modality = {}
    source = "missing"
    for path in modality_paths:
        if path.is_file():
            modality = json.loads(path.read_text(encoding="utf-8"))
            source = str(path)
            break

    state = modality.get("state", {}) if isinstance(modality, dict) else {}
    fields = {
        name: {
            "original_key": spec.get("original_key"),
            "start": spec.get("start"),
            "end": spec.get("end"),
        }
        for name, spec in state.items()
        if isinstance(spec, dict)
    }
    return {
        "source": source,
        "fields": fields,
        "has_base_pose": "base_position" in fields and "base_rotation" in fields,
        "has_eef_pose": "end_effector_position_relative" in fields
        and "end_effector_rotation_relative" in fields,
        "has_gripper": "gripper_qpos" in fields,
        "has_object_pose": any("object" in name for name in fields),
        "has_fixture_joint_or_contact": any(
            key in name for name in fields for key in ("joint", "contact", "fixture", "drawer")
        ),
    }


def _audit_reward_done(dataset_path: Path, episode_indices: tuple[int, ...]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA future label source audit requires pyarrow.") from exc

    row_count = 0
    reward_values: set[float] = set()
    done_true = 0
    success_done = 0
    non_success_done = 0
    h10_labeled = 0
    h10_positive = 0
    h10_negative = 0

    for episode_index in episode_indices:
        parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        if not parquet_path.is_file():
            continue
        table = pq.read_table(parquet_path, columns=["next.reward", "next.done"])
        data = table.to_pydict()
        rewards = [float(value) for value in data.get("next.reward", ())]
        dones = [bool(value) for value in data.get("next.done", ())]
        row_count += len(rewards)
        reward_values.update(rewards)
        for reward, done in zip(rewards, dones):
            if not done:
                continue
            done_true += 1
            if reward > 0:
                success_done += 1
            else:
                non_success_done += 1

        for anchor in range(len(rewards)):
            label = _failure_risk_label_from_window(rewards, dones, anchor, horizon=10)
            if label is None:
                continue
            h10_labeled += 1
            if label > 0:
                h10_positive += 1
            else:
                h10_negative += 1

    return {
        "sampled_row_count": row_count,
        "reward_unique_values_preview": tuple(sorted(reward_values))[:16],
        "reward_is_sparse_0_1_on_sample": reward_values.issubset({0.0, 1.0}) if reward_values else False,
        "done_true_count": done_true,
        "success_done_count": success_done,
        "non_success_done_count": non_success_done,
        "failure_risk_h10_labeled_count": h10_labeled,
        "failure_risk_h10_positive_count": h10_positive,
        "failure_risk_h10_negative_count": h10_negative,
        "failure_risk_h10_coverage": (h10_labeled / row_count) if row_count else 0.0,
    }


def _failure_risk_label_from_window(
    rewards: list[float],
    dones: list[bool],
    anchor: int,
    horizon: int,
) -> float | None:
    for reward, done in zip(rewards[anchor + 1 : anchor + 1 + horizon], dones[anchor + 1 : anchor + 1 + horizon]):
        if not done:
            continue
        return 0.0 if reward > 0 else 1.0
    return None


def _binary_phi_coefficient(pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    n11 = sum(x == 1 and y == 1 for x, y in pairs)
    n10 = sum(x == 1 and y == 0 for x, y in pairs)
    n01 = sum(x == 0 and y == 1 for x, y in pairs)
    n00 = sum(x == 0 and y == 0 for x, y in pairs)
    left = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if left <= 0:
        return None
    return ((n11 * n00) - (n10 * n01)) / (left ** 0.5)


def _ordered_joint_state_layout(xml_root: ET.Element) -> list[dict[str, Any]]:
    qpos_index = 1
    qvel_index = 1
    entries = []
    for joint in xml_root.iter("joint"):
        joint_type = joint.attrib.get("type", "hinge")
        nq = 7 if joint_type == "free" else 4 if joint_type == "ball" else 1
        nv = 6 if joint_type == "free" else 3 if joint_type == "ball" else 1
        range_text = joint.attrib.get("range", "")
        range_value = tuple(float(value) for value in range_text.split()) if range_text else ()
        entries.append(
            {
                "name": joint.attrib["name"],
                "type": joint_type,
                "nq": nq,
                "nv": nv,
                "qpos_index": qpos_index,
                "qvel_index": qvel_index,
                "range": range_value,
            }
        )
        qpos_index += nq
        qvel_index += nv
    return entries


def _build_opendrawer_episode_mapping_sample(
    states_path: Path,
    episode_index: int,
    drawer_qpos_index: int,
    open_denominator: float,
) -> MoWAOpenDrawerEpisodeMappingSample:
    states = np.load(states_path, allow_pickle=True)["states"]
    drawer_qpos = states[:, drawer_qpos_index]
    normalized_open_max = max(0.0, float((-drawer_qpos.min()) / open_denominator)) if open_denominator > 0 else 0.0
    return MoWAOpenDrawerEpisodeMappingSample(
        episode_index=episode_index,
        state_length=int(states.shape[0]),
        drawer_qpos_first=float(drawer_qpos[0]),
        drawer_qpos_last=float(drawer_qpos[-1]),
        drawer_qpos_min=float(drawer_qpos.min()),
        drawer_qpos_max=float(drawer_qpos.max()),
        drawer_normalized_open_max=normalized_open_max,
    )


def _build_opendrawer_subgoal_episode_smoke(
    parquet_path: Path,
    states_path: Path,
    horizon: int,
) -> MoWAOpenDrawerSubgoalEpisodeSmoke:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA OpenDrawer subgoal data gate smoke requires pyarrow.") from exc

    table = pq.read_table(parquet_path, columns=["next.reward", "next.done"])
    data = table.to_pydict()
    rewards = [float(value) for value in data["next.reward"]]
    dones = [bool(value) for value in data["next.done"]]
    drawer_qpos_index, open_denominator = _resolve_opendrawer_drawer_progress_layout(states_path)
    states = np.load(states_path, allow_pickle=True)["states"]
    drawer_progress = (-states[:, drawer_qpos_index]) / open_denominator if open_denominator > 0 else np.zeros(len(states))

    positive_count = 0
    negative_count = 0
    for anchor in range(len(drawer_progress)):
        future_window = drawer_progress[anchor + 1 : anchor + 1 + horizon]
        if np.any(future_window >= 0.95):
            positive_count += 1
        else:
            negative_count += 1

    terminal_success = bool(dones[-1] and rewards[-1] > 0) if rewards else False
    state_success = bool(np.max(drawer_progress) >= 0.95) if len(drawer_progress) else False
    if terminal_success and state_success:
        alignment_status = "aligned_success"
    elif terminal_success and not state_success:
        alignment_status = "terminal_success_without_state_success"
    elif (not terminal_success) and state_success:
        alignment_status = "state_success_without_terminal_success"
    else:
        alignment_status = "aligned_failure"

    return MoWAOpenDrawerSubgoalEpisodeSmoke(
        episode_index=int(parquet_path.stem.split("_")[-1]),
        row_count=len(rewards),
        state_length=int(states.shape[0]),
        terminal_reward=float(rewards[-1]) if rewards else 0.0,
        terminal_done=bool(dones[-1]) if dones else False,
        terminal_success_proxy=terminal_success,
        drawer_progress_max=float(np.max(drawer_progress)) if len(drawer_progress) else 0.0,
        drawer_progress_terminal=float(drawer_progress[-1]) if len(drawer_progress) else 0.0,
        state_success_proxy=state_success,
        anchor_positive_count=positive_count,
        anchor_negative_count=negative_count,
        anchor_positive_rate=(positive_count / (positive_count + negative_count))
        if (positive_count + negative_count)
        else 0.0,
        alignment_status=alignment_status,
    )


def _resolve_opendrawer_drawer_progress_layout(states_path: Path) -> tuple[int, float]:
    episode_dir = states_path.parent
    ep_meta_path = episode_dir / "ep_meta.json"
    model_path = episode_dir / "model.xml.gz"
    if not ep_meta_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"OpenDrawer episode extras missing for {episode_dir}")

    ep_meta = json.loads(ep_meta_path.read_text(encoding="utf-8"))
    fixture_ref_value = ep_meta["fixture_refs"]["drawer"]
    xml_root = ET.fromstring(gzip.decompress(model_path.read_bytes()))
    ordered_joints = _ordered_joint_state_layout(xml_root)
    drawer_joint_name = f"{fixture_ref_value}_slidejoint"
    drawer_entry = next(entry for entry in ordered_joints if entry["name"] == drawer_joint_name)
    joint_range = tuple(float(value) for value in drawer_entry["range"])
    open_denominator = abs(joint_range[0]) * 0.55 / 2
    return int(drawer_entry["qpos_index"]), open_denominator


def _build_opendrawer_future_label_episode_sidecar(
    parquet_path: Path,
    states_path: Path,
    failure_risk_horizon: int,
    subgoal_horizon: int,
    readiness_horizon: int,
    readiness_progress_delta: float,
    schema_contract: dict[str, Any],
) -> MoWAOpenDrawerFutureLabelEpisodeSidecar:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA OpenDrawer future-label sidecar smoke requires pyarrow.") from exc

    table = pq.read_table(parquet_path, columns=["next.reward", "next.done", "frame_index"])
    data = table.to_pydict()
    rewards = [float(value) for value in data["next.reward"]]
    dones = [bool(value) for value in data["next.done"]]
    frame_indices = [int(value) for value in data.get("frame_index", range(len(rewards)))]
    drawer_qpos_index, open_denominator = _resolve_opendrawer_drawer_progress_layout(states_path)
    drawer_qvel_index = _resolve_opendrawer_drawer_qvel_index(states_path)
    states = np.load(states_path, allow_pickle=True)["states"]

    row_count = min(len(rewards), len(dones), len(frame_indices), int(states.shape[0]))
    rewards = rewards[:row_count]
    dones = dones[:row_count]
    frame_indices = frame_indices[:row_count]
    states = states[:row_count]

    drawer_progress = (-states[:, drawer_qpos_index]) / open_denominator if open_denominator > 0 else np.zeros(row_count)
    drawer_qvel = states[:, drawer_qvel_index]

    terminal_success = bool(dones[-1] and rewards[-1] > 0) if row_count else False
    state_success = bool(np.max(drawer_progress) >= 0.95) if row_count else False
    if terminal_success and state_success:
        alignment_status = "aligned_success"
    elif terminal_success and not state_success:
        alignment_status = "terminal_success_without_state_success"
    elif (not terminal_success) and state_success:
        alignment_status = "state_success_without_terminal_success"
    else:
        alignment_status = "aligned_failure"

    steps = tuple(
        _build_opendrawer_step_label(
            timestep=timestep,
            frame_index=frame_indices[timestep],
            drawer_progress=drawer_progress,
            drawer_qvel=drawer_qvel,
            rewards=rewards,
            dones=dones,
            failure_risk_horizon=failure_risk_horizon,
            subgoal_horizon=subgoal_horizon,
            readiness_horizon=readiness_horizon,
            readiness_progress_delta=readiness_progress_delta,
        )
        for timestep in range(row_count)
    )

    return MoWAOpenDrawerFutureLabelEpisodeSidecar(
        episode_index=int(parquet_path.stem.split("_")[-1]),
        row_count=row_count,
        state_length=int(states.shape[0]),
        schema_contract=schema_contract,
        terminal_success_proxy=terminal_success,
        state_success_proxy=state_success,
        alignment_status=alignment_status,
        steps=steps,
    )


def _build_opendrawer_schema_contract(readiness_progress_delta: float) -> dict[str, Any]:
    return {
        "subgoal": {
            "subgoal_id": OPENDRAWER_SUBGOAL_ID,
            "schema_version": OPENDRAWER_SUBGOAL_SCHEMA_VERSION,
            "completion_predicate": "drawer_progress >= 0.95",
            "completion_threshold": OPENDRAWER_SUBGOAL_COMPLETION_THRESHOLD,
            "source_chain": (
                "ep_meta.fixture_refs.drawer -> model.xml.gz -> slidejoint -> states.npz qpos",
            ),
        },
        "readiness": {
            "schema_version": OPENDRAWER_READINESS_SCHEMA_VERSION,
            "readiness_type": OPENDRAWER_READINESS_TYPE,
            "readiness_target": OPENDRAWER_READINESS_TARGET,
            "ready_predicate": "future drawer progress gain >= readiness_progress_delta",
            "manipulation_onset_predicate": "future drawer progress delta >= readiness_progress_delta",
            "threshold_version": "readiness_progress_delta_v1",
            "readiness_progress_delta": readiness_progress_delta,
        },
    }


def _resolve_opendrawer_drawer_qvel_index(states_path: Path) -> int:
    episode_dir = states_path.parent
    ep_meta_path = episode_dir / "ep_meta.json"
    model_path = episode_dir / "model.xml.gz"
    if not ep_meta_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"OpenDrawer episode extras missing for {episode_dir}")

    ep_meta = json.loads(ep_meta_path.read_text(encoding="utf-8"))
    fixture_ref_value = ep_meta["fixture_refs"]["drawer"]
    xml_root = ET.fromstring(gzip.decompress(model_path.read_bytes()))
    ordered_joints = _ordered_joint_state_layout(xml_root)
    drawer_joint_name = f"{fixture_ref_value}_slidejoint"
    drawer_entry = next(entry for entry in ordered_joints if entry["name"] == drawer_joint_name)
    inferred_nq = sum(int(entry["nq"]) for entry in ordered_joints)
    drawer_qvel_index = 1 + inferred_nq + int(drawer_entry["qvel_index"]) - 1
    return drawer_qvel_index


def _build_opendrawer_step_label(
    timestep: int,
    frame_index: int,
    drawer_progress: np.ndarray,
    drawer_qvel: np.ndarray,
    rewards: list[float],
    dones: list[bool],
    failure_risk_horizon: int,
    subgoal_horizon: int,
    readiness_horizon: int,
    readiness_progress_delta: float,
) -> MoWAOpenDrawerStepLabel:
    current_progress = float(drawer_progress[timestep])
    current_qvel = float(drawer_qvel[timestep])
    completed_now = current_progress >= OPENDRAWER_SUBGOAL_COMPLETION_THRESHOLD

    failure_risk = _failure_risk_label_from_window(
        rewards=rewards,
        dones=dones,
        anchor=timestep,
        horizon=failure_risk_horizon,
    )
    failure_risk_mask = failure_risk is None

    subgoal_future = drawer_progress[timestep + 1 : timestep + 1 + subgoal_horizon]
    subgoal_mask = completed_now
    subgoal_feasibility = 0.0
    subgoal_time_to_completion = -1
    if not subgoal_mask:
        completion_indices = np.flatnonzero(subgoal_future >= OPENDRAWER_SUBGOAL_COMPLETION_THRESHOLD)
        if completion_indices.size > 0:
            subgoal_feasibility = 1.0
            subgoal_time_to_completion = int(completion_indices[0]) + 1

    readiness_future = drawer_progress[timestep + 1 : timestep + 1 + readiness_horizon]
    readiness_mask = completed_now or readiness_future.size == 0
    manipulation_readiness = 0.0
    readiness_event_time = -1
    readiness_future_gain = 0.0
    if not readiness_mask:
        future_gain = readiness_future - current_progress
        readiness_future_gain = float(np.max(future_gain)) if future_gain.size else 0.0
        event_indices = np.flatnonzero(future_gain >= readiness_progress_delta)
        if event_indices.size > 0:
            manipulation_readiness = 1.0
            readiness_event_time = int(event_indices[0]) + 1

    return MoWAOpenDrawerStepLabel(
        timestep=timestep,
        frame_index=frame_index,
        drawer_progress=current_progress,
        drawer_qvel=current_qvel,
        predicates={
            "open_drawer_completed": completed_now,
            "future_open_drawer_completion_within_horizon": subgoal_feasibility > 0.0,
            "future_open_event_within_readiness_window": manipulation_readiness > 0.0,
        },
        labels={
            "failure_risk": float(failure_risk) if failure_risk is not None else 0.0,
            "subgoal_id": OPENDRAWER_SUBGOAL_ID if not subgoal_mask else "none",
            "subgoal_schema_version": OPENDRAWER_SUBGOAL_SCHEMA_VERSION,
            "subgoal_completion_predicate": "drawer_progress >= 0.95",
            "subgoal_completion_threshold": OPENDRAWER_SUBGOAL_COMPLETION_THRESHOLD,
            "subgoal_feasibility": subgoal_feasibility,
            "time_to_subgoal": subgoal_time_to_completion,
            "manipulation_readiness": manipulation_readiness,
            "readiness_schema_version": OPENDRAWER_READINESS_SCHEMA_VERSION,
            "readiness_type": OPENDRAWER_READINESS_TYPE if not readiness_mask else "none",
            "readiness_target": OPENDRAWER_READINESS_TARGET if not readiness_mask else "none",
            "ready_predicate": "future drawer progress gain >= readiness_progress_delta",
            "manipulation_onset_predicate": "future drawer progress delta >= readiness_progress_delta",
            "threshold_version": "readiness_progress_delta_v1",
            "readiness_event_time": readiness_event_time,
        },
        masks={
            "failure_risk": not failure_risk_mask,
            "subgoal_feasibility": not subgoal_mask,
            "manipulation_readiness": not readiness_mask,
        },
        debug={
            "failure_risk_source": "reward_done_horizon_proxy" if not failure_risk_mask else "masked_no_done_in_window",
            "subgoal_source": "drawer_progress_predicate",
            "subgoal_schema_version": OPENDRAWER_SUBGOAL_SCHEMA_VERSION,
            "readiness_source": (
                "future_drawer_progress_delta_proxy"
                if not readiness_mask
                else "masked_completed_or_short_future_window"
            ),
            "readiness_schema_version": OPENDRAWER_READINESS_SCHEMA_VERSION,
            "readiness_future_gain": readiness_future_gain,
            "readiness_progress_delta": readiness_progress_delta,
            "profile_status": "measured",
        },
    )


def _audit_extras(dataset_path: Path, episode_indices: tuple[int, ...]) -> dict[str, Any]:
    extras_dir = dataset_path / "extras"
    sampled = []
    states_shapes = []
    model_joint_name_previews = []
    model_keyword_hits: set[str] = set()

    for episode_index in episode_indices:
        episode_dir = extras_dir / f"episode_{episode_index:06d}"
        states_path = episode_dir / "states.npz"
        ep_meta_path = episode_dir / "ep_meta.json"
        model_path = episode_dir / "model.xml.gz"
        entry: dict[str, Any] = {
            "episode_index": episode_index,
            "episode_dir_exists": episode_dir.is_dir(),
            "states_npz_exists": states_path.is_file(),
            "ep_meta_exists": ep_meta_path.is_file(),
            "model_xml_gz_exists": model_path.is_file(),
        }
        if states_path.is_file():
            import numpy as np

            with np.load(states_path, allow_pickle=True) as npz:
                keys = tuple(npz.keys())
                states = npz["states"] if "states" in npz else None
                entry["states_keys"] = keys
                entry["states_shape"] = tuple(int(dim) for dim in states.shape) if states is not None else None
                if states is not None:
                    states_shapes.append(entry["states_shape"])
        if ep_meta_path.is_file():
            ep_meta = json.loads(ep_meta_path.read_text(encoding="utf-8"))
            fixtures = ep_meta.get("fixtures", {}) if isinstance(ep_meta, dict) else {}
            objects = ep_meta.get("object_cfgs", ()) if isinstance(ep_meta, dict) else ()
            entry["fixture_class_preview"] = tuple(
                sorted({fixture.get("cls", "unknown") for fixture in fixtures.values() if isinstance(fixture, dict)})
            )[:16]
            entry["object_name_preview"] = tuple(
                obj.get("name", "unknown") for obj in objects[:8] if isinstance(obj, dict)
            )
        if model_path.is_file():
            model_scan = _scan_model_xml(model_path)
            joint_names = model_scan["joint_names"]
            entry["model_joint_name_preview"] = joint_names[:24]
            entry["model_keyword_hits"] = model_scan["keyword_hits"]
            model_joint_name_previews.extend(joint_names[:24])
            model_keyword_hits.update(model_scan["keyword_hits"])
        sampled.append(entry)

    return {
        "extras_dir_exists": extras_dir.is_dir(),
        "sampled_extras": tuple(sampled),
        "states_shape_preview": tuple(states_shapes),
        "model_joint_name_preview": tuple(model_joint_name_previews[:32]),
        "model_keyword_hits": tuple(sorted(model_keyword_hits)),
        "has_simulator_state_sequence": any(item.get("states_npz_exists") for item in sampled),
        "has_ep_meta": any(item.get("ep_meta_exists") for item in sampled),
        "has_model_xml": any(item.get("model_xml_gz_exists") for item in sampled),
        "has_task_fixture_or_object_in_model_xml": bool(model_keyword_hits),
    }


def _scan_model_xml(model_path: Path) -> dict[str, Any]:
    text = gzip.decompress(model_path.read_bytes()).decode("utf-8", errors="ignore")
    names = []
    for marker in ('<joint name="', "<joint name='"):
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            name_start = idx + len(marker)
            quote = marker[-1]
            name_end = text.find(quote, name_start)
            if name_end < 0:
                break
            names.append(text[name_start:name_end])
            start = name_end + 1
    lower_text = text.lower()
    keywords = (
        "drawer",
        "cabinet",
        "fridge",
        "toaster",
        "coffee",
        "sink",
        "faucet",
        "mug",
        "handle",
        "contact",
    )
    return {
        "joint_names": tuple(names),
        "keyword_hits": tuple(keyword for keyword in keywords if keyword in lower_text),
    }


def _audit_task_source(task: str, repo_root: Path) -> dict[str, Any]:
    source_root = repo_root / "playground/Code/robocasa365/robocasa/environments/kitchen/atomic"
    matches = []
    if source_root.is_dir():
        for path in sorted(source_root.glob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if f"class {task}" in text:
                snippet = _extract_class_snippet(text, task)
                matches.append(
                    {
                        "source_path": str(path),
                        "has_check_success": "_check_success" in snippet,
                        "mentions_reward_success": "success" in snippet,
                        "predicate_keywords": tuple(
                            keyword
                            for keyword in ("get_door_state", "check_obj_fixture_contact", "get_handle_state", "knob_joints")
                            if keyword in snippet
                        ),
                    }
                )
    return {
        "task_class_found": bool(matches),
        "matches": tuple(matches),
    }


def _extract_class_snippet(text: str, task: str) -> str:
    marker = f"class {task}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_class = text.find("\nclass ", start + len(marker))
    if next_class < 0:
        next_class = len(text)
    return text[start:next_class]


def _build_label_readiness(
    columns: tuple[str, ...],
    reward_done_audit: dict[str, Any],
    extras_audit: dict[str, Any],
    state_schema: dict[str, Any],
) -> dict[str, str]:
    column_set = set(columns)
    reward_done_available = {"next.reward", "next.done"}.issubset(column_set)
    failure_ready = (
        "candidate_for_data_gate"
        if reward_done_available and reward_done_audit["reward_is_sparse_0_1_on_sample"]
        else "blocked_by_reward_done_distribution_or_missing_fields"
    )
    subgoal_ready = (
        "blocked_pending_task_subgoal_schema_and_predicate_mapping"
        if extras_audit["has_simulator_state_sequence"] and extras_audit["has_model_xml"]
        else "blocked_by_missing_simulator_state_or_model_xml"
    )
    manipulation_ready = (
        "blocked_pending_readiness_predicate_and_threshold_calibration"
        if state_schema["has_base_pose"] and state_schema["has_eef_pose"] and state_schema["has_gripper"]
        else "blocked_by_missing_state_semantics"
    )
    return {
        "failure_risk": failure_ready,
        "subgoal_feasibility": subgoal_ready,
        "manipulation_readiness": manipulation_ready,
    }
