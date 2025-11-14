#!/usr/bin/env python3
"""Compatibility shim that defers to the packaged Auto Boost CLI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from auto_boost.cli import main


if __name__ == "__main__":
    main()
