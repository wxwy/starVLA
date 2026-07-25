"""StarFlow-VLA LIBERO eval smoke 前置检查。"""

import subprocess
import unittest
from pathlib import Path

from omegaconf import OmegaConf


STAGE1_CONFIG = Path("configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml")
RUN_POLICY_SERVER = Path("examples/LIBERO/eval_files/run_policy_server.sh")
EVAL_LIBERO = Path("examples/LIBERO/eval_files/eval_libero.sh")
RUN_EVAL_REGRESSION = Path("examples/LIBERO/eval_files/run_starflow_eval_regression.sh")


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = []
    for path in checkpoint_dir.glob("steps_*"):
        if path.is_dir():
            step_text = path.name.removeprefix("steps_")
        elif path.name.endswith("_pytorch_model.pt"):
            step_text = path.name.removeprefix("steps_").removesuffix("_pytorch_model.pt")
        else:
            continue
        if step_text.isdigit():
            candidates.append((int(step_text), path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


class StarFlowEvalPreflightTest(unittest.TestCase):
    def test_eval_shell_scripts_are_syntax_valid(self):
        for script in (RUN_POLICY_SERVER, EVAL_LIBERO, RUN_EVAL_REGRESSION):
            result = subprocess.run(["bash", "-n", str(script)], check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_eval_shell_supports_quick_regression_mode(self):
        script_text = EVAL_LIBERO.read_text(encoding="utf-8")
        self.assertIn("MAX_TASKS=${MAX_TASKS:-}", script_text)
        self.assertIn("--args.max-tasks", script_text)

        regression_text = RUN_EVAL_REGRESSION.read_text(encoding="utf-8")
        self.assertIn("run_policy_server.sh", regression_text)
        self.assertIn("eval_libero.sh", regression_text)
        self.assertIn("eval_report.json", regression_text)

    def test_stage1_checkpoint_has_mapping_before_eval(self):
        cfg = OmegaConf.load(STAGE1_CONFIG)
        run_dir = Path(cfg.run_root_dir) / cfg.run_id
        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            self.skipTest(f"P0 checkpoint directory not found: {checkpoint_dir}")

        checkpoint = _latest_checkpoint(checkpoint_dir)
        if checkpoint is None:
            self.skipTest(f"No P0 checkpoint found in: {checkpoint_dir}")

        if checkpoint.suffix:
            mapping_path = checkpoint.with_name(f"{checkpoint.name}.starflow_mapping.json")
        else:
            mapping_path = checkpoint / "starflow_mapping.json"
        self.assertTrue(mapping_path.exists(), f"Missing starflow mapping sidecar: {mapping_path}")


if __name__ == "__main__":
    unittest.main()
