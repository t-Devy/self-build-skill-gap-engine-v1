

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "skill_gaps.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
META_PATH = ARTIFACTS_DIR / "meta.json"
REPORT_PATH = ARTIFACTS_DIR / "report.md"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"



FEATURE_COLS = [
    "safety_gap",
    "tools_gap",
    "theory_gap",
    "communication_gap",
    "career_gap",
]

TARGET_COL = "priority_skill"


ALLOWED_LABELS = [
    "safety",
    "tools",
    "theory",
    "communication",
    "career",
]
