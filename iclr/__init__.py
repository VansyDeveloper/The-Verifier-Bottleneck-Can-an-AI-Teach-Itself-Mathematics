"""ICLR follow-up; the frozen Stage4 interpreter and scorer live in legacy/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))
