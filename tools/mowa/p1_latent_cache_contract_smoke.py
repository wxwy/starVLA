"""P1 latent cache contract smoke compatibility entrypoint."""

from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    script_path = Path(__file__).with_name("latent_cache_contract_smoke.py")
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
