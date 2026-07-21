"""EXT-B1 Phase 1: proxy server — StarVLA websocket protocol <-> official GR00T N1.5.

Runs in the ext_b1_groot venv. Serves the StarVLA websocket policy protocol
(deployment/model_server/tools/websocket_policy_server.py) while inference is
performed by the OFFICIAL Gr00tPolicy (Isaac-GR00T) — official processor,
official unnormalization, official inference path, nothing re-implemented.

Client payload (payload_style="groot"):
    examples[i] = {
        "image": [left, right, wrist]  # uint8 (256,256,3), RAW — no client resize
        "lang": str,
        "state_raw": (16,) float       # raw 16-d, NO sin/cos
    }
Returns {"actions": (B, 16, 12)} in env space (already unnormalized by the
official modality transform; NO second unnormalization is applied).

Usage:
    playground/.venvs/ext_b1_groot/bin/python tools/mowa/mowa_external_groot_n15_proxy_server.py \
        --model-path /disk/rl/models/robocasa365_baselines/e000_b1_gr00t_n1_5/gr00t_n1-5/multitask_learning/checkpoint-120000 \
        --port 6791
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "playground" / "Code" / "Isaac-GR00T"))
sys.path.insert(0, str(_REPO))

from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402

from deployment.model_server.tools.websocket_policy_server import (  # noqa: E402
    WebsocketPolicyServer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ext_b1_proxy")

# Official action keys in the official order (data_config.py:653-659) — identical
# to the client's ACTION_SLICES order.
OFFICIAL_ACTION_KEYS = [
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
]
ACTION_KEY_DIMS = [3, 3, 1, 4, 1]
STATE_KEY_ORDER = [
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]


class ExternalGrootN15Adapter:
    """Thin adapter: StarVLA predict_action(examples) -> official Gr00tPolicy.get_action."""

    def __init__(self, model_path: str, embodiment_tag: str, data_config_name: str) -> None:
        data_config = DATA_CONFIG_MAP[data_config_name]
        self._action_keys = list(data_config.action_keys)
        assert self._action_keys == OFFICIAL_ACTION_KEYS, (
            f"official action_keys changed: {self._action_keys}"
        )
        self._policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=embodiment_tag,
            denoising_steps=4,
        )
        self._model_path = model_path

    def predict_action(self, examples: list[dict], **kwargs) -> dict:
        batch = len(examples)
        videos = {k: [] for k in ("left", "right", "wrist")}
        langs, states = [], []
        for i, ex in enumerate(examples):
            imgs = ex["image"]
            if len(imgs) != 3:
                raise ValueError(f"example {i}: expected 3 views, got {len(imgs)}")
            for name, img in zip(("left", "right", "wrist"), imgs):
                arr = np.asarray(img)
                if arr.shape != (256, 256, 3) or arr.dtype != np.uint8:
                    raise ValueError(
                        f"example {i} view {name}: expected uint8 (256,256,3), "
                        f"got {arr.dtype} {arr.shape} (client must send RAW frames)"
                    )
                videos[name].append(arr)
            langs.append(ex["lang"])
            s = np.asarray(ex["state_raw"], dtype=np.float64).ravel()
            if s.shape != (16,):
                raise ValueError(f"example {i}: state_raw must be (16,), got {s.shape}")
            states.append(s)

        obs = {
            "video.robot0_agentview_left": np.stack(videos["left"])[:, None],    # (B,1,256,256,3)
            "video.robot0_agentview_right": np.stack(videos["right"])[:, None],
            "video.robot0_eye_in_hand": np.stack(videos["wrist"])[:, None],
            "state.base_position": np.stack(states)[:, None, 0:3],
            "state.base_rotation": np.stack(states)[:, None, 3:7],
            "state.end_effector_position_relative": np.stack(states)[:, None, 7:10],
            "state.end_effector_rotation_relative": np.stack(states)[:, None, 10:14],
            "state.gripper_qpos": np.stack(states)[:, None, 14:16],
            "annotation.human.task_description": np.array(langs)[:, None],
        }
        out = self._policy.get_action(obs)  # per-action-key dict, env space

        parts = []
        for key, dim_k in zip(self._action_keys, ACTION_KEY_DIMS):
            if key not in out:
                raise KeyError(f"official get_action missing key {key}; got {list(out.keys())}")
            v = np.asarray(out[key])
            if v.ndim == 2:
                v = v[None]
            assert v.shape[0] == batch and v.shape[-1] == dim_k, (
                f"{key}: expected (B,T,{dim_k}), got {v.shape}"
            )
            parts.append(v.astype(np.float64))
        actions = np.concatenate(parts, axis=-1)  # (B,16,12)
        assert actions.shape[1] == 16 and actions.shape[2] == 12, actions.shape
        return {"actions": actions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6791)
    args = parser.parse_args()

    adapter = ExternalGrootN15Adapter(args.model_path, args.embodiment_tag, args.data_config)
    metadata = {
        "env": "ext_b1_groot_n15_proxy",
        "ckpt_path": args.model_path,
        "config_overrides": [],
        "action_chunk_size": 16,
        "action_keys": OFFICIAL_ACTION_KEYS,
        "state_keys": STATE_KEY_ORDER,
        "available_unnorm_keys": ["external_groot_n15_raw"],
        "default_unnorm_key": "external_groot_n15_raw",
    }
    logger.info("starting EXT-B1 proxy on %s:%d", args.host, args.port)
    server = WebsocketPolicyServer(adapter, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
