"""StarFlow-VLA 训练启动脚本前置检查。"""

import subprocess
import unittest
from pathlib import Path


TRAIN_READY = Path("examples/LIBERO/train_files/run_starflow_train_ready.sh")


class StarFlowTrainReadyTest(unittest.TestCase):
    def test_train_ready_shell_is_syntax_valid(self):
        result = subprocess.run(["bash", "-n", str(TRAIN_READY)], check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_train_ready_shell_contains_known_good_defaults(self):
        script_text = TRAIN_READY.read_text(encoding="utf-8")
        self.assertIn("--config_yaml configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml", script_text)
        self.assertIn("MASTER_ADDR", script_text)
        self.assertIn("WORLD_SIZE", script_text)
        self.assertIn("--datasets.vla_data.per_device_batch_size \"${PER_DEVICE_BATCH_SIZE}\"", script_text)
        self.assertIn("--trainer.gradient_accumulation_steps \"${GRADIENT_ACCUMULATION_STEPS}\"", script_text)
        self.assertIn("--trainer.save_checkpoint_as_directory True", script_text)
        self.assertIn("accelerate launch", script_text)


if __name__ == "__main__":
    unittest.main()
