"""Check MoWA final report template exists and contains required sections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TEMPLATE = Path("docs_zh/mowa/11_final_report_known_limitations_template.md")
OUTPUT = Path("docs_zh/mowa/mowa_final_report_template_smoke.json")
REQUIRED_HEADINGS = (
    "## 1. 实验范围",
    "## 2. 已完成证据",
    "## 3. Known Limitations",
    "## 4. 未解决问题",
    "## 5. 后续动作",
    "## 6. 附录",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA final report template smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_final_report_template_smoke(args.repo_root, args.template)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_final_report_template_smoke(
    repo_root: Path | str,
    template_path: Path = TEMPLATE,
) -> dict[str, Any]:
    root = Path(repo_root)
    full_path = root / template_path
    text = full_path.read_text(encoding="utf-8") if full_path.is_file() else ""
    checks = {
        "template_exists": full_path.is_file(),
        "all_required_headings_present": all(heading in text for heading in REQUIRED_HEADINGS),
    }
    return {
        "stage": "M6",
        "task_id": "M6-002",
        "training_started": False,
        "checks": checks,
        "template": str(template_path),
        "observed": {
            "required_heading_count": len(REQUIRED_HEADINGS),
            "present_heading_count": sum(1 for heading in REQUIRED_HEADINGS if heading in text),
        },
        "unresolved_items": [
            "This smoke checks template structure only.",
            "Final report content still depends on future measured experiments.",
        ],
        "go_no_go": (
            "TBD: final report template smoke passed"
            if all(checks.values())
            else "No-Go: final report template smoke failed"
        ),
    }


if __name__ == "__main__":
    main()
