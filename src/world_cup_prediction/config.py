from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CLUB_RESULTS_CSV = DATA_DIR / "club_results.csv"
NATIONAL_RESULTS_CSV = DATA_DIR / "national_results.csv"
TEAMS_MAPPING_JSON = DATA_DIR / "teams_mapping.json"
PRETRAINED_CLUBS_PATH = ARTIFACTS_DIR / "pretrained_clubs.pth"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "world_cup_model.pth"
TOURNAMENT_STATE_PATH = ARTIFACTS_DIR / "tournament_state.pkl"

SEQ_LEN = 5
NUM_FEATURES = 5
BATCH_SIZE = 32
