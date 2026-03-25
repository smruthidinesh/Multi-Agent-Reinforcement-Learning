from pathlib import Path
import sys


# Make the src package importable in Vercel's runtime.
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl.visualization.web_server import app  # noqa: E402,F401
