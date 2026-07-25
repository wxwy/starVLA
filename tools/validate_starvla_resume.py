#!/usr/bin/env python
"""Validate StarVLA resume path without entering the training loop."""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.train_starvla import (
    VLATrainer,
    accelerator,
    logger,
    prepare_data,
    setup_directories,
    setup_optimizer_and_scheduler,
)
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(clipargs))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    cfg = wrap_config(cfg)

    output_dir = setup_directories(cfg=cfg)
    model = build_framework(cfg)
    dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    logger.info(f"Resume validation succeeded at completed_steps={trainer.completed_steps}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
