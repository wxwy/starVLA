import dataclasses
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import torch
import tqdm
import tyro


_ORIG_TORCH_LOAD = torch.load


def _torch_load_compat(*args, **kwargs):
    # LIBERO init_states are trusted pickle files rather than pure model weights.
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_compat

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient
from examples.LIBERO.eval_files.starflow_eval_report import (
    build_eval_report,
    infer_failure_category,
    load_eval_metadata,
    load_eval_report,
    write_eval_report,
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.
    task_ids: str = ""  # Comma-separated task IDs to evaluate, e.g. "0,2,5". Empty = run all (bounded by max_tasks).
    replan_interval: int | None = None  # None = use full action chunk; 1/2/4/... = replan cadence in env steps
    resume_eval: bool = False  # If true, continue from an existing eval_report.json in video_out_path
    report_filename: str = ""  # Custom eval report filename (e.g. "eval_report_task_0.json"). Empty = default.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    eval_metadata = load_eval_metadata(args.pretrained_path)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        replan_interval=args.replan_interval,
    )

    # 手检：验证 server 服务的 ckpt 与预期一致
    server_ckpt = client_model._server_metadata.get("ckpt_path", "")
    expected_ckpt = str(pathlib.Path(args.pretrained_path).resolve())
    server_ckpt_resolved = str(pathlib.Path(server_ckpt).resolve())
    if server_ckpt_resolved != expected_ckpt:
        raise RuntimeError(
            f"❌ Server checkpoint mismatch!\n"
            f"  Server serves: {server_ckpt}\n"
            f"  Expected:      {expected_ckpt}\n"
            f"  Check --port, another eval might be connected to wrong server."
        )
    logging.info("✅ Server ckpt match: %s", server_ckpt)

    # Optional smoke-test cap (still useful for quick verification with -1 = full run).
    n_eval_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    # Resume from existing report when explicitly requested.
    total_episodes, total_successes = 0, 0
    episode_records = []
    completed_episodes: set[tuple[int, int]] = set()
    resumed_task_rollup: dict[int, tuple[int, int]] = {}
    report_filename = args.report_filename or "eval_report.json"
    if args.resume_eval:
        existing_report = load_eval_report(args.video_out_path, filename=report_filename)
        if existing_report is not None:
            existing_ckpt = existing_report.get("checkpoint_path")
            if existing_ckpt and pathlib.Path(existing_ckpt) != pathlib.Path(args.pretrained_path):
                raise RuntimeError(
                    f"resume_eval checkpoint mismatch: report={existing_ckpt} current={args.pretrained_path}"
                )
            if existing_report.get("task_suite_name") != args.task_suite_name:
                raise RuntimeError(
                    f"resume_eval task_suite mismatch: report={existing_report.get('task_suite_name')} "
                    f"current={args.task_suite_name}"
                )
            episode_records = list(existing_report.get("episodes", []))
            total_episodes = int(existing_report.get("total_episodes", len(episode_records)))
            total_successes = int(existing_report.get("total_successes", 0))
            for record in episode_records:
                task_id = int(record["task_id"])
                episode_idx = int(record["episode_idx"])
                completed_episodes.add((task_id, episode_idx))
                episodes_done, successes_done = resumed_task_rollup.get(task_id, (0, 0))
                resumed_task_rollup[task_id] = (
                    episodes_done + 1,
                    successes_done + int(bool(record.get("success"))),
                )
            logging.info(
                "Resuming eval from %s: total_episodes=%d total_successes=%d completed_pairs=%d",
                pathlib.Path(args.video_out_path) / report_filename,
                total_episodes,
                total_successes,
                len(completed_episodes),
            )

    # Determine which task IDs to evaluate
    if args.task_ids:
        selected_task_ids = [int(t.strip()) for t in args.task_ids.split(",") if t.strip()]
        logging.info(f"Using explicit task_ids: {selected_task_ids}")
    else:
        selected_task_ids = list(range(n_eval_tasks))

    # Start evaluation
    for task_id in tqdm.tqdm(selected_task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = resumed_task_rollup.get(task_id, (0, 0))
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            if (task_id, episode_idx) in completed_episodes:
                continue
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []
            done = False
            runtime_error = None

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")
            try:
                while t < max_steps + args.num_steps_wait:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        wait_step_start = time.perf_counter()
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        wait_step_elapsed = time.perf_counter() - wait_step_start
                        if t == 0:
                            logging.info("Warmup env.step_sec=%.4f", wait_step_elapsed)
                        t += 1
                        continue

                    obs_prepare_start = time.perf_counter()
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    observation = {  #
                        "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                        "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                        "observation.state": np.expand_dims(state, axis=0),
                        "instruction": [str(task_description)],
                    }

                    # align key with model API --> two images provided here --> check training
                    example_dict = {
                        "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                        "lang": observation["instruction"][0],
                        "state": observation["observation.state"],  # (1, state_dim) matching training sample
                    }
                    obs_prepare_elapsed = time.perf_counter() - obs_prepare_start

                    infer_start = time.perf_counter()
                    response = client_model.step(example=example_dict, step=step)
                    infer_elapsed = time.perf_counter() - infer_start

                    raw_action = response["raw_action"]
                    response_timings = response.get("timings", {})

                    world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                    rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                    open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                    gripper = _binarize_gripper_open(open_gripper)

                    if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                        logging.warning(
                            f"Unexpected action sizes: "
                            f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                            f"Falling back to LIBERO_DUMMY_ACTION."
                        )
                        raise ValueError(
                            f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                            f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                        )

                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
                    full_actions.append(delta_action)

                    env_step_start = time.perf_counter()
                    obs, reward, done, info = env.step(delta_action.tolist())
                    env_step_elapsed = time.perf_counter() - env_step_start
                    if response_timings.get("cache_refresh"):
                        chunk_timings = response_timings.get("chunk_request") or {}
                        logging.info(
                            "Timing step=%d obs_prepare=%.4fs infer_total=%.4fs resize=%.4fs "
                            "client_pack=%.4fs client_roundtrip=%.4fs client_unpack=%.4fs client_total=%.4fs "
                            "server_total=%.4fs "
                            "framework_total=%.4fs prepare_inputs=%.4fs build_qwen_inputs=%.4fs "
                            "qwen_forward=%.4fs gather_action_tokens=%.4fs action_head=%.4fs "
                            "to_numpy=%.4fs unnorm=%.4fs env_step=%.4fs",
                            step,
                            obs_prepare_elapsed,
                            infer_elapsed,
                            chunk_timings.get("resize_sec") or 0.0,
                            chunk_timings.get("client_pack_and_queue_sec") or 0.0,
                            chunk_timings.get("client_server_roundtrip_sec") or 0.0,
                            chunk_timings.get("client_unpack_sec") or 0.0,
                            chunk_timings.get("client_total_call_sec") or 0.0,
                            chunk_timings.get("server_total_sec") or 0.0,
                            chunk_timings.get("framework_total_sec") or 0.0,
                            chunk_timings.get("framework_prepare_inputs_sec") or 0.0,
                            chunk_timings.get("framework_build_qwen_inputs_sec") or 0.0,
                            chunk_timings.get("framework_qwen_forward_sec") or 0.0,
                            chunk_timings.get("framework_gather_action_tokens_sec") or 0.0,
                            chunk_timings.get("framework_action_head_sec") or 0.0,
                            chunk_timings.get("framework_to_numpy_sec") or 0.0,
                            chunk_timings.get("server_unnorm_sec") or 0.0,
                            env_step_elapsed,
                        )
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    step += 1
            except Exception as exc:
                runtime_error = str(exc)
                logging.exception("LIBERO rollout failed", exc_info=exc)

            task_episodes += 1
            total_episodes += 1

            # Capture inference timing from the last chunk
            chunk_timings = client_model._last_chunk_timings or {}

            # Save report before video encoding so smoke metadata is preserved even
            # if EGL/video cleanup stalls during process teardown.
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4"
            episode_records.append(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode_idx": episode_idx,
                    "success": bool(done),
                    "failure_category": infer_failure_category(success=bool(done), runtime_error=runtime_error),
                    "runtime_error": runtime_error,
                    "steps_executed": step,
                    "video_path": str(video_path) if video_path is not None else None,
                    "chunk_timings": chunk_timings,
                }
            )
            interim_report = build_eval_report(
                args=dataclasses.asdict(args),
                metadata=eval_metadata,
                total_episodes=total_episodes,
                total_successes=total_successes,
                episode_records=episode_records,
            )
            write_eval_report(args.video_out_path, interim_report, filename=report_filename)

            if replay_images:
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            else:
                video_path = None

            if full_actions:
                full_actions = np.stack(full_actions)
                del full_actions

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    report = build_eval_report(
        args=dataclasses.asdict(args),
        metadata=eval_metadata,
        total_episodes=total_episodes,
        total_successes=total_successes,
        episode_records=episode_records,
    )
    report_path = write_eval_report(args.video_out_path, report, filename=report_filename)
    logging.info(f"Eval report saved to: {report_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
