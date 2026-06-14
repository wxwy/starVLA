"""StarFlow-VLA future token 消融配置与 mapping 测试。"""

import unittest
from copy import deepcopy

from omegaconf import OmegaConf

from starVLA.model.modules.starflow_vla.mapping import build_starflow_mapping


CONFIG_PATH = "configs/starflow_vla/stage3_future_token_ablation.yaml"


class StarFlowFutureTokenVariantsTest(unittest.TestCase):
    def test_all_future_token_variants_emit_mapping(self):
        cfg = OmegaConf.load(CONFIG_PATH)
        values = list(cfg.ablation.num_target_vision_tokens_values)
        self.assertEqual(values, [0, 8, 16, 32, 64])

        for value in values:
            with self.subTest(num_target_vision_tokens=value):
                variant = deepcopy(cfg)
                variant.framework.action_model.num_target_vision_tokens = value
                mapping = build_starflow_mapping(variant)
                self.assertEqual(mapping["num_target_vision_tokens"], value)
                self.assertEqual(mapping["adapter_mode"], "future_token_cross_dit")
                self.assertEqual(mapping["state_mode"], "discretized_instruction")
                self.assertEqual(mapping["action_head"], "LayerwiseFM")


if __name__ == "__main__":
    unittest.main()
