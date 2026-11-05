"""The tests import the package from this checkout and the torch references (``reference``) and kernel-test helpers
(``kernels/``) from this directory."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE / "kernels"), str(HERE / "unit")]
