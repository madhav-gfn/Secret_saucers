import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = PROJECT_ROOT

def find_data_dir():
    
    candidates = []

    env_path = os.environ.get("REDROB_DATA_DIR")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([
        PROJECT_ROOT / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge",
        PROJECT_ROOT / "India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge",
        PROJECT_ROOT / "India_runs_data_and_ai_challenge",
        PROJECT_ROOT / "Documentation",
        PROJECT_ROOT,
    ])

    for path in candidates:
        if (path / "job_description.txt").exists():
            return path

    raise FileNotFoundError(
        "Could not find Redrob data. Put job_description.txt and candidates.jsonl "
        "inside RUNS/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge "
        "or set REDROB_DATA_DIR."
    )


DATA_DIR = find_data_dir()
SAMPLE_CANDIDATES_PATH = DATA_DIR / "sample_candidates.json"
FULL_CANDIDATES_PATH = DATA_DIR / "candidates.jsonl"
JD_PATH = DATA_DIR / "job_description.txt"

# ──────────────────────────────────────────────
# Feature Weights (must sum to 1.0)
# ──────────────────────────────────────────────
# Design rationale:
#   - semantic_similarity is reduced from the previous 40% to 18%.
#     Embeddings are good for broad retrieval but cannot distinguish
#     "built a ranking system" from "took a course on ranking."
#   - retrieval_relevance is the #1 engineered feature (15%) because
#     the JD's ideal candidate is explicitly a retrieval/ranking/search person.
#   - product_execution (12%) rewards "shipped" > "studied" — JD's core value.
#   - evaluation_systems (10%) is rare and highly valued by the JD.
#   - No single feature exceeds 18%.

FEATURE_WEIGHTS = {
    "semantic":              0.18,
    "retrieval_relevance":   0.15,
    "product_execution":     0.12,
    "evaluation_systems":    0.10,
    "title_relevance":       0.08,
    "experience_alignment":  0.08,
    "career_stability":      0.08,
    "activity":              0.06,
    "availability":          0.05,
    "vector_search":         0.04,
    "nlp_llm":               0.03,
    "consistency":           0.03,
}

# Penalty deduction caps (applied as score -= penalty_value * cap)
PENALTY_CAPS = {
    "research":        0.08,
    "specialization":  0.06,
    "consulting":      0.06,
}

# Hard-skill matching weight (folded into semantic for Stage 2)
HARD_SKILL_BLEND = 0.35  # Within the semantic slot, 35% hard-skill + 65% embedding

# Trap Penalties (legacy — kept for Stage 1 hard filters)
NOTICE_PERIOD_TRAP_DAYS = 90  # Relaxed from 60; soft penalty handles 45-90 range
MIN_EXPERIENCE_YEARS = 2      # Relaxed from 3; soft penalty handles nuance
