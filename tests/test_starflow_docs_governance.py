"""StarFlow-VLA 文档与 patch 管理治理测试。"""

import re
import unittest
from pathlib import Path


DOC_DIR = Path("docs_zh/starflow_vla")
README = DOC_DIR / "README.md"
PATCH_MANIFEST = DOC_DIR / "PATCH_MANIFEST.md"

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


if __name__ == "__main__":
    unittest.main()
