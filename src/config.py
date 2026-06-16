from pathlib import Path
BASE_DIR = Path(r"c:\Users\devgu\Downloads\[PUB] India_runs_data_and_ai_challenge")
# Project root (repo root)
# Using a repo-relative path so the pipeline uses the dataset included in this workspace
# madhav's base directory
# BASE_DIR = Path(__file__).resolve().parents[1]

# Data directory inside the repo
DATA_DIR = BASE_DIR / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "sample_candidates.json"
FULL_CANDIDATES_PATH = DATA_DIR / "candidates.jsonl"
JD_PATH = DATA_DIR / "job_description.txt"

# Synthesized Pipeline Weights
WEIGHTS = {
    "semantic": 0.40,
    "hard_skills": 0.30,
    "behavioral": 0.30
}

# Trap Penalties
NOTICE_PERIOD_TRAP_DAYS = 60
MIN_EXPERIENCE_YEARS = 3
