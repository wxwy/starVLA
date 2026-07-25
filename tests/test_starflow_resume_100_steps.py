"""StarFlow-VLA 100-step resume consistency smoke。"""

import copy
import shutil
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate.utils import DistributedType
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.starflow_vla.mapping import save_starflow_checkpoint_mapping
from starVLA.training.train_starvla import VLATrainer, setup_optimizer_and_scheduler


CHECKPOINT_DIR = Path("playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1")


def _clone_batch(batch):
    if isinstance(batch, torch.Tensor):
        return batch.detach().clone()
    if isinstance(batch, list):
        return [_clone_batch(item) for item in batch]
    if isinstance(batch, tuple):
        return tuple(_clone_batch(item) for item in batch)
    if isinstance(batch, dict):
        return {key: _clone_batch(value) for key, value in batch.items()}
    return copy.deepcopy(batch)


def _move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    if isinstance(batch, list):
        return [_move_to_cuda(item) for item in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_cuda(item) for item in batch)
    if isinstance(batch, dict):
        return {key: _move_to_cuda(value) for key, value in batch.items()}
    return batch


class DummyAccelerator:
    distributed_type = DistributedType.NO
    num_processes = 1
    process_index = 0
    is_main_process = True
    is_local_main_process = True
    sync_gradients = True
    gradient_accumulation_steps = 1

    @staticmethod
    def accumulate(_model):
        return nullcontext()

    @staticmethod
    def backward(loss):
        loss.backward()

    @staticmethod
    def clip_grad_norm_(parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def wait_for_everyone():
        return None

    @staticmethod
    def print(*args, **kwargs):
        return None


class StarFlowResume100StepsTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not CHECKPOINT_DIR.exists():
            self.skipTest(f"Checkpoint not found: {CHECKPOINT_DIR}")

    def _build_cfg(self, output_dir: Path):
        cfg = OmegaConf.load(CHECKPOINT_DIR / "config.yaml")
        cfg = apply_config_compat(cfg)
        cfg.output_dir = str(output_dir)
        cfg.run_root_dir = str(output_dir.parent)
        cfg.run_id = output_dir.name
        cfg.wandb_project = "disabled"
        cfg.wandb_entity = "disabled"
        cfg.trainer.enable_local_checkpoint_staging = False
        cfg.trainer.max_train_steps = 101
        cfg.trainer.eval_interval = 1000000
        cfg.trainer.save_interval = 1000000
        return cfg

    def _build_trainer(self, output_dir: Path, checkpoint_path: Path, *, load_training_state: bool):
        cfg = self._build_cfg(output_dir)
        model = build_framework(cfg)
        dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)
        accelerator = DummyAccelerator()
        trainer = VLATrainer(
            cfg=cfg,
            model=model,
            vla_train_dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
        )
        trainer.model = trainer.load_pretrained_backbones(
            trainer.model,
            str(checkpoint_path),
            reload_modules=None,
            preferred_format=getattr(cfg.trainer, "save_format", "pt"),
        )
        trainer.model = trainer.freeze_backbones(trainer.model, freeze_modules=cfg.trainer.freeze_modules)
        trainer.model = trainer.model.cuda()
        if load_training_state:
            trainer._load_lightweight_training_state(checkpoint_path)
        trainer.model.train()
        return trainer

    def _run_steps(self, trainer, batch, steps: int):
        losses = []
        for index in range(steps):
            metrics = trainer._train_step(_clone_batch(batch))
            trainer.completed_steps += 1
            losses.append(float(metrics["action_dit_loss"]))
            if (index + 1) % 10 == 0:
                print(
                    f"[resume-smoke] completed_steps={trainer.completed_steps} "
                    f"phase_step={index + 1}/{steps} loss={losses[-1]:.8f}",
                    flush=True,
                )
        return losses

    def test_resume_after_50_steps_matches_continuous_100_steps(self):
        tmp_parent = Path("playground/tmp_resume_smoke")
        tmp_parent.mkdir(parents=True, exist_ok=True)
        tmp_root = Path(tempfile.mkdtemp(prefix="resume_100_", dir=tmp_parent))
        try:
            bootstrap_dir = tmp_root / "bootstrap_run"
            continuous_dir = tmp_root / "continuous_run"
            split_dir = tmp_root / "split_run"
            resumed_dir = tmp_root / "resumed_run"

            set_seed(42)
            bootstrap_trainer = self._build_trainer(bootstrap_dir, CHECKPOINT_DIR, load_training_state=False)
            batch = _move_to_cuda(_clone_batch(next(iter(bootstrap_trainer.vla_train_dataloader))))
            bootstrap_losses = self._run_steps(bootstrap_trainer, batch, 1)
            self.assertEqual(len(bootstrap_losses), 1)
            bootstrap_checkpoint = bootstrap_dir / "checkpoints" / f"steps_{bootstrap_trainer.completed_steps}"
            bootstrap_trainer._save_lightweight_directory_checkpoint(
                bootstrap_checkpoint,
                getattr(bootstrap_trainer.config.trainer, "save_format", "pt"),
            )
            for helper_name in ("config.yaml", "config.full.yaml", "dataset_statistics.json"):
                source = CHECKPOINT_DIR / helper_name
                if source.exists():
                    shutil.copy2(source, bootstrap_checkpoint / helper_name)
            save_starflow_checkpoint_mapping(bootstrap_checkpoint, bootstrap_trainer.config)
            del bootstrap_trainer
            torch.cuda.empty_cache()

            set_seed(42)
            continuous_trainer = self._build_trainer(continuous_dir, bootstrap_checkpoint, load_training_state=True)
            continuous_losses = self._run_steps(continuous_trainer, batch, 100)
            del continuous_trainer
            torch.cuda.empty_cache()

            set_seed(42)
            split_trainer = self._build_trainer(split_dir, bootstrap_checkpoint, load_training_state=True)
            split_losses_first = self._run_steps(split_trainer, batch, 50)
            self.assertEqual(split_trainer.completed_steps, 51)

            resume_checkpoint = split_dir / "checkpoints" / f"steps_{split_trainer.completed_steps}"
            split_trainer._save_lightweight_directory_checkpoint(
                resume_checkpoint,
                getattr(split_trainer.config.trainer, "save_format", "pt"),
            )
            for helper_name in ("config.yaml", "config.full.yaml", "dataset_statistics.json"):
                source = CHECKPOINT_DIR / helper_name
                if source.exists():
                    shutil.copy2(source, resume_checkpoint / helper_name)
            save_starflow_checkpoint_mapping(resume_checkpoint, split_trainer.config)

            resumed_trainer = self._build_trainer(resumed_dir, resume_checkpoint, load_training_state=True)
            self.assertEqual(resumed_trainer.completed_steps, 51)
            resumed_losses_second = self._run_steps(resumed_trainer, batch, 50)
            del split_trainer
            del resumed_trainer
            torch.cuda.empty_cache()
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.assertEqual(len(continuous_losses), 100)
        self.assertEqual(len(split_losses_first), 50)
        self.assertEqual(len(resumed_losses_second), 50)

        continuous_tail = continuous_losses[50:]
        resumed_tail = resumed_losses_second
        relative_deltas = [
            abs(resumed - continuous) / max(abs(continuous), 1e-8)
            for continuous, resumed in zip(continuous_tail, resumed_tail)
        ]
        max_relative_delta = max(relative_deltas)
        mean_relative_delta = sum(relative_deltas) / len(relative_deltas)

        self.assertLess(
            max_relative_delta,
            0.01,
            msg=f"resume 50->100 max relative delta too large: max={max_relative_delta:.6f}, mean={mean_relative_delta:.6f}",
        )


if __name__ == "__main__":
    unittest.main()
