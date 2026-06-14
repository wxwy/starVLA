"""StarFlow-VLA LIBERO batch schema smoke 测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


CONFIG_PATH = Path("configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml")


class StarFlowLIBEROBatchTest(unittest.TestCase):
    def test_stage1_config_points_to_libero_7dof_batch(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        data_root = Path(cfg.datasets.vla_data.data_root_dir)

        self.assertEqual(cfg.framework.name, "StarFlowVLA")
        self.assertEqual(cfg.framework.action_model.action_dim, 7)
        self.assertTrue(cfg.datasets.vla_data.include_state)
        self.assertEqual(cfg.datasets.vla_data.data_mix, "libero_goal")

        if not data_root.exists():
            self.skipTest(f"LIBERO data root not found: {data_root}")

        from starVLA.dataloader import build_dataloader

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            dataloader = build_dataloader(cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
            batch = next(iter(dataloader))

        self.assertIsInstance(batch, list)
        self.assertGreater(len(batch), 0)
        sample = batch[0]
        for key in ("image", "lang", "state", "action"):
            self.assertIn(key, sample)

        action = np.asarray(sample["action"])
        state = np.asarray(sample["state"])
        self.assertEqual(action.shape[-1], 7)
        self.assertEqual(state.shape[-1], 7)
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue(np.isfinite(state).all())


if __name__ == "__main__":
    unittest.main()
