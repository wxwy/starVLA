# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
        config_overrides: Optional[List[str]] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)
        self._config_overrides = list(config_overrides or [])
        overall_start = time.perf_counter()

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        stage_start = time.perf_counter()
        framework = baseframework.from_pretrained(
            self._ckpt_path,
            config_overrides=self._config_overrides,
        )
        logging.info(
            "PolicyServerWrapper: baseframework.from_pretrained finished in %.2fs",
            time.perf_counter() - stage_start,
        )
        if use_bf16:
            stage_start = time.perf_counter()
            framework = framework.to(torch.bfloat16)
            logging.info(
                "PolicyServerWrapper: cast to bfloat16 finished in %.2fs",
                time.perf_counter() - stage_start,
            )
        stage_start = time.perf_counter()
        framework = framework.to(device).eval()
        logging.info(
            "PolicyServerWrapper: move to %s + eval finished in %.2fs",
            device,
            time.perf_counter() - stage_start,
        )
        self._framework = framework

        self._model_cfg = getattr(framework, "config", None)
        if hasattr(framework, "action_horizon"):
            self._action_chunk_size = int(framework.action_horizon)
        else:
            action_model_cfg = self._model_cfg.framework.action_model
            if hasattr(action_model_cfg, "action_horizon"):
                self._action_chunk_size = int(action_model_cfg.action_horizon)
            elif hasattr(action_model_cfg, "future_action_window_size"):
                self._action_chunk_size = int(action_model_cfg.future_action_window_size) + 1
            else:
                raise ValueError(
                    "PolicyServerWrapper: no action_horizon or future_action_window_size found "
                    f"in override-applied model config for {self._ckpt_path}"
                )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())
        logging.info(
            "PolicyServerWrapper: discovered available_unnorm_keys=%s",
            self._available_unnorm_keys,
        )

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            stage_start = time.perf_counter()
            default_proc = self._get_processor(unnorm_key)
            logging.info(
                "PolicyServerWrapper: PolicyNormProcessor init finished in %.2fs",
                time.perf_counter() - stage_start,
            )
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )
        logging.info(
            "PolicyServerWrapper: fully initialized in %.2fs",
            time.perf_counter() - overall_start,
        )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "config_overrides": self._config_overrides,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        overall_start = time.perf_counter()
        # 验证输入是否包含 state（仅记录一次或低频）
        if examples and isinstance(examples, list) and len(examples) > 0:
            ex0 = examples[0]
            input_keys = sorted(ex0.keys()) if isinstance(ex0, dict) else []
            state_info = "NO_STATE"
            if "state" in input_keys and ex0["state"] is not None:
                try:
                    state_arr = np.asarray(ex0["state"])
                    state_info = f"state_shape={state_arr.shape}, state_dtype={state_arr.dtype}"
                except Exception:
                    state_info = "state_present_but_not_array"
            logging.info(
                "PolicyServerWrapper.predict_action: input_keys=%s, %s",
                input_keys,
                state_info,
            )

        proc = self._get_processor(effective_key)

        framework_start = time.perf_counter()
        out = self._framework.predict_action(examples=examples, **kwargs)
        framework_done = time.perf_counter()
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm_start = time.perf_counter()
        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        unnorm_done = time.perf_counter()
        framework_timings = out.get("timings", {}) if isinstance(out, dict) else {}
        return {
            "actions": unnorm,
            "timings": {
                "server_total_sec": unnorm_done - overall_start,
                "framework_sec": framework_done - framework_start,
                "unnorm_sec": unnorm_done - unnorm_start,
                **framework_timings,
            },
        }
