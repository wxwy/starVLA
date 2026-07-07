# MoWA TODO

- 补 `E-001` long-training command draft / launch entrypoint，并把 baseline / MoWA 两套 4090 profile（`bs=2`、`ga=16`、`warmup=500`）接到正式启动前 preview。
- P1 fake-encoder 版 latent cache writer / validator / loader 已完成；下一步接真实 Wan adapter，再补 E-003 config preview / train dry-run；不得直接启动 E-003 训练。
- 明确 `class_mapping_status` 从 `Data Gate` 到确认态的收口条件，并同步到 readiness / matrix。
- 使用更强 checkpoint 重新执行 `E-006` rollout，对 bridge 干预拿到有信息量的对比证据。
