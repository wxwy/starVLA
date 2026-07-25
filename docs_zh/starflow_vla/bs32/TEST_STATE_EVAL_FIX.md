# LIBERO 评测 state 传入修复记录

## 问题

2026-07-16 排查发现：所有 LIBERO 正式评测中，`eval_libero.py` 没有把机器人 proprioceptive state 传给模型。

- 训练配置（`configs/starflow_vla/ablations/future_tokens_*.yaml`、`continuous_head.yaml`、`stage2_mlp_baseline.yaml`）均设置 `datasets.vla_data.include_state: true`。
- 但 `examples/LIBERO/eval_files/eval_libero.py` 构造的 `example_dict` 只包含 `image` 和 `lang`，丢弃了已经计算好的 `observation.state`。
- 导致训练-评测不一致（train-eval mismatch）：模型训练时依赖 state，评测时却收不到 state。

## 修复

文件：`examples/LIBERO/eval_files/eval_libero.py`

修改前：

```python
example_dict = {
    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
    "lang": observation["instruction"][0],
}
```

修改后：

```python
example_dict = {
    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
    "lang": observation["instruction"][0],
    "state": observation["observation.state"],  # (1, state_dim) matching training sample
}
```

`state` 形状保持 `(1, state_dim)`，与 dataloader 训练样本一致。

## 受影响实验

以下已推送的 LIBERO 实验报告可能需要在修复后重新评测：

- `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848`
- `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849`
- `P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854`
- `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010`
- `P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260703_0928`
- `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021`

## 验证

- [x] `python -m py_compile examples/LIBERO/eval_files/eval_libero.py` 通过
- [ ] 实际跑一个 task 的评测，确认 server 收到 `state` 且动作输出正常
- [ ] 对比修复前后同一 checkpoint 的 success rate

## 备注

- `deployment/model_server/policy_wrapper.py` 和框架 `StarFlowVLA` / `Qwen_PI_v3` 已支持 `state`，无需改动。
- wrapper 脚本（`nohup_eval.sh`、`nohup_eval_pool.sh`、`eval_pool_manager.py`）无需改动。
