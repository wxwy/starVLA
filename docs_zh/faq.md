# 常见问题

<details>
<summary><b>Q: 为什么不把预处理放在 dataloader 里？</b></summary>

A: 我们做过性能分析：数据预处理耗时不到 1%。将其保留在 Framework 内部是可以接受的，并且允许模型特定的灵活处理。

</details>

<details>
<summary><b>Q: 可以用 Qwen2.5-VL 以外的骨干网络吗？</b></summary>

A: 可以。实现新的视觉 + 语言模块并在 Framework 中组合它们；任何其他现有模型都可以替换进来。而且，由于框架直接处理原始动作数据，替换非常容易。

</details>

<details>
<summary><b>Q: 为什么没有视觉塔的抽象接口？</b></summary>

A: 我们认为 VLM 将成为基础模型，并且本身就拥有原生的视觉塔。

</details>

<details>
<summary><b>Q: 可以通过终端覆盖或添加参数吗？</b></summary>

A: 可以。我们使用 `OmegaConf.load(args.config_yaml)` 作为单一配置入口；独立调试也使用 `args.config_yaml`。参数可以有意冗余；你可以通过 CLI 自由添加或覆盖它们。

示例：
```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml  \
  --num_processes 8 \
  starVLA/training/train_internvla.py \
  --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml \
  --framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct \
  --framework.action_model.new_module ${module_name}
```

⚠️：`framework.action_model.new_module` 仅添加到全局配置；具体行为取决于你的框架。

</details>

<details>
<summary><b>Q: 可以通过参数冻结 VLM 吗？</b></summary>

A: 可以。StarVLA 使用正则 / 名称列表来控制冻结。示例：
```
--trainer.freeze_modules "qwen_vl_interface.model.model.visual,dino_encoder"
```
提示：可以先 `print(your_model)` 查看模块的相对路径，然后以逗号分隔列出。
（实现在 `TrainerUtils.freeze_backbones` 中。）

</details>

<details>
<summary><b>Q: 可以为不同模块设置不同的学习率吗？</b></summary>

A: 可以，starVLA 同样使用 name: value 字典来控制学习率分组。配置示例：
```yaml
trainer:
  learning_rate:
    base: 1e-05      # 其他模块
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
```
（同样在 `trainer_tools.build_param_lr_groups` 中引用。）

</details>

<details>
<summary><b>Q: 可以从检查点恢复训练吗？</b></summary>

A: 可以，某种程度上可以。在 `config.yaml` 中指定最新的检查点路径，例如：
```yaml
trainer:
  pretrained_checkpoint: path_to_steps_10000.pt
  reload_modules: "action_model"
```
`reload_modules` 为空表示完整加载所有模型。但 starVLA 不保存优化器状态，因为这需要大量内存/磁盘空间而收益有限。

</details>

<details>
<summary><b>Q: 如何用更小的 VLM 训练？</b></summary>

```bash
    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
      --machine_rank $SLURM_PROCID \
      --num_machines $SLURM_NNODES \
      --num_processes=${TOTAL_GPUS} \
      starVLA/training/train_starvla.py \
      --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml \
      --framework.framework_py QwenGR00T \
      --framework.qwenvl.base_vlm microsoft/Florence-2-large \
      --run_root_dir ${run_root_dir} \
      --run_id ${run_id} \
      --wandb_project your_project \
      --wandb_entity your_name
```

注意：为确保与已发布检查点的更好兼容性，我们继续使用 `--framework.qwenvl`。此参数将在下个版本中统一。

</details>
