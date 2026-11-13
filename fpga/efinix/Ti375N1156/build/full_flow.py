#!/usr/bin/env python3
"""Regenerate Ti375N1156_HSB (interface + IP).

Synth / P&R / program in the Efinity GUI.

Prerequisites:
  source <EFINITY_HOME>/bin/setup.sh
  Close the Efinity GUI before regenerating the project and IP.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BOARD_DIR = SCRIPT_DIR.parent
BUILD_INTERFACE = SCRIPT_DIR / "build_interface.py"
IP_GEN = SCRIPT_DIR / "ip_gen.py"
PROJECT_XML = BOARD_DIR / "Ti375N1156_HSB" / "N1156_HSB.xml"


def require_env() -> None:
    missing = [
        name for name in ("EFINITY_HOME", "EFXPT_HOME", "EFXIPM_HOME") if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(
            "Missing environment variable(s): "
            + ", ".join(missing)
            + ". Source <EFINITY_HOME>/bin/setup.sh first."
        )


def run(cmd: list[str], *, cwd: Path) -> None:
    print(f"\n==> {' '.join(cmd)}  (cwd={cwd})")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}"
        ) from None


def main() -> None:
    require_env()
    run([sys.executable, str(BUILD_INTERFACE)], cwd=SCRIPT_DIR)
    run([sys.executable, str(IP_GEN)], cwd=SCRIPT_DIR)
    if not PROJECT_XML.is_file():
        raise SystemExit(f"Expected project file missing after build: {PROJECT_XML}")
    print(f"Ready: {PROJECT_XML}")
    print("\nDone. Open N1156_HSB.xml in Efinity for synth / P&R / program.")


if __name__ == "__main__":
    main()
