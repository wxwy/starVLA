"""StarFlow-VLA 文档与 patch 管理治理测试。"""

import json
import re
import unittest
from pathlib import Path


DOC_DIR = Path("docs_zh/starflow_vla")
README = DOC_DIR / "README.md"
PATCH_MANIFEST = DOC_DIR / "PATCH_MANIFEST.md"
MODULE_MAPPING = DOC_DIR / "MODULE_MAPPING.md"
P0_IMPLEMENTATION_PLAN = DOC_DIR / "P0_IMPLEMENTATION_PLAN.md"
DESIGN_FREEZE_CHECK = DOC_DIR / "DESIGN_FREEZE_CHECK.md"
STAGE1_CONFIG = Path("configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml")
STAGE1_MAPPING = Path(
    "playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1/starflow_mapping.json"
)

REQUIRED_DOCS = [
    DOC_DIR / "DESIGN.md",
    DOC_DIR / "BASELINE_VERSION.md",
    DOC_DIR / "UPSTREAM_COMPATIBILITY.md",
    DOC_DIR / "MODULE_MAPPING.md",
    DOC_DIR / "PATCH_MANIFEST.md",
    DOC_DIR / "EXPERIMENT_MATRIX.md",
    DOC_DIR / "EVAL_SMOKE.md",
    DOC_DIR / "IMPLEMENTATION_LOG.md",
]

TRACKED_ARTIFACTS = [
    "starVLA/model/framework/VLM4A/StarFlowVLA.py",
    "starVLA/model/modules/starflow_vla/mapping.py",
    "configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml",
    "configs/starflow_vla/stage2_mlp_baseline.yaml",
    "configs/starflow_vla/stage3_future_token_ablation.yaml",
    "tests/test_starflow_vla_reuse.py",
    "tests/test_starflow_libero_batch.py",
    "tests/test_starflow_checkpoint_mapping.py",
    "tests/test_starflow_eval_preflight.py",
    "tests/test_starflow_docs_governance.py",
    "tests/test_starflow_future_token_variants.py",
]


class StarFlowDocsGovernanceTest(unittest.TestCase):
    def test_required_docs_exist(self):
        missing = [str(path) for path in REQUIRED_DOCS if not path.exists()]
        self.assertEqual(missing, [])

    def test_readme_links_resolve(self):
        text = README.read_text(encoding="utf-8")
        links = re.findall(r"\]\((\./[^)]+)\)", text)
        missing = []
        for link in links:
            target = (DOC_DIR / link.removeprefix("./")).resolve()
            if not target.exists():
                missing.append(link)
        self.assertEqual(missing, [])

    def test_patch_manifest_mentions_tracked_artifacts(self):
        text = PATCH_MANIFEST.read_text(encoding="utf-8")
        missing = [artifact for artifact in TRACKED_ARTIFACTS if artifact not in text]
        self.assertEqual(missing, [])

    def test_p0_guardrails_do_not_require_perceiver_flowcondition_or_14d_mask(self):
        design_freeze_text = DESIGN_FREEZE_CHECK.read_text(encoding="utf-8")
        self.assertIn("PerceiverAdapter、显式 `FlowCondition` runtime、`14D action_mask` 不作为 P0 必选实现。", design_freeze_text)

        implementation_plan_text = P0_IMPLEMENTATION_PLAN.read_text(encoding="utf-8")
        self.assertIn("P0 不包含 PerceiverAdapter、显式 FlowCondition runtime、14D action_mask。", implementation_plan_text)

        module_mapping_text = MODULE_MAPPING.read_text(encoding="utf-8")
        self.assertIn("| Explicit FlowCondition runtime | 可选 dataclass / wrapper | P2，不阻断 P0 |", module_mapping_text)
        self.assertIn("| PerceiverAdapter | 可选 token compressor | P2，不阻断 P0 |", module_mapping_text)
        self.assertIn("| 7/14DoF action mask | `max_action_dim=14 + action_mask + masked loss` | P1，不阻断 P0 |", module_mapping_text)

        stage1_config_text = STAGE1_CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("perceiver_enabled: true", stage1_config_text)
        self.assertNotIn("flow_condition_runtime: true", stage1_config_text)
        self.assertNotIn("max_action_dim: 14", stage1_config_text)
        self.assertNotIn("action_mask", stage1_config_text)

        mapping_payload = json.loads(STAGE1_MAPPING.read_text(encoding="utf-8"))
        self.assertFalse(mapping_payload["perceiver_enabled"])
        self.assertFalse(mapping_payload["flow_condition_runtime"])


if __name__ == "__main__":
    unittest.main()
