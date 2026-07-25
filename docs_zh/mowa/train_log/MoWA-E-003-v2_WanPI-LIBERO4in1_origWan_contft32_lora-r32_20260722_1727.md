# MoWA E-003-v2 LIBERO4in1 训练记录

> run id：`MoWA-E-003-v2_WanPI-LIBERO4in1_origWan_contft32_lora-r32_20260722_1727`
> 启动时间：2026-07-22 17:40 CST
> 配置：`configs/mowa/mowa_e003_v2_wanpi_libero4in1_contft32_lora.yaml`

## 运行结果

- 8×RTX 4090、bf16、per-device batch=2、GA=2、global batch=32。
- `libero_4in1_wanpi` 数据集、H0 输入已通过 data-flow validation 2/2；Wan 输入 `T=9=0 history+1 current+8 future`。
- 训练在 step 266 因 DataLoader worker bus error 退出；不是模型 loss/梯度异常。

## 根因与修复（2026-07-22）

- tmux `train` 报错：`RuntimeError: unable to write to file </torch_...>: No space left on device (28)`，随后 rank3 报 `DataLoader worker ... Bus error`；`/dev/shm` 容量为 10 GiB。
- 根因：工作区未提交改动将 `instruction_text_latent` 再次传入 worker 侧 `MoWALatentCacheDataset`，使 UMT5 `text_embeds` 随 sample 跨 worker/rank 的共享内存队列传输，回归了此前已修复的 shm 问题。
- 修复：`starVLA/dataloader/gr00t_lerobot/datasets.py` 恢复 `instruction_text_latent=None`。worker 仅传轻量 `lang`，WanPI 在每个训练 rank 内从同一 instruction cache 查表补齐 embedding；保留 LIBERO 的可配置 anchor video key 兼容逻辑。
- 验证：`.venv/bin/python -m pytest tests/mowa/test_train_starvla_window_latent_integration.py -q`，**5 passed**。

## 后续启动要求

- 使用新的 run id 从 scratch 重启，不恢复该失败 run。
- 首次 2-step data-flow 后观察 `/dev/shm`；至少运行至超过 step 266，再确认不再出现 `torch_*` 写入失败或 DataLoader bus error。
- 运行时保存的 `config.full.yaml`、W&B URL、实际 git SHA 与 checkpoint 必须写入对应 checklist。
