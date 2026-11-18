"""Compatibility entry point; implementation and parameters live in kwon.scan_check."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kwon.scan_check import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
