import sys
import time
from pathlib import Path

# Import torch first on Windows to avoid DLL initialization conflicts with sklearn/scipy/faiss
import torch

# Ensure the project root is on sys.path so `from src.xxx` works
# regardless of how this script is invoked (any machine, any cwd).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.config import SAMPLE_CANDIDATES_PATH, FULL_CANDIDATES_PATH, JD_PATH, BASE_DIR
from src.stage2_ensemble import EnsembleMatcher
from src.stage1_filter import stream_and_filter_candidates
from src.stage1b_tfidf import tfidf_prefilter
from src.stage3_scorecard import rank_candidates
from src.stage4_explanation import (
    generate_debug_csv,
    generate_submission_csv,
    generate_top20_debug_csv,
    print_feature_diagnostics,
)

def run_pipeline(use_sample=False):
    data_path = SAMPLE_CANDIDATES_PATH if use_sample else FULL_CANDIDATES_PATH
    
    start_time = time.time()
    
    print(f"--- Starting Improved Pipeline with data: {data_path.name} ---")
    
    # 1. STAGE 1: Viability Pruning (relaxed — soft penalties handle nuance)
    print("\n--- STAGE 1: Fast Pruning (Hard Filters Only) ---")
    s1_candidates = list(stream_and_filter_candidates(data_path))
    print(f"Stage 1 Filtering: Retained {len(s1_candidates)} candidates.")
    
    # 1.5. STAGE 1.5: TF-IDF Pre-filter (reduce pool for fast embedding)
    print("\n--- STAGE 1.5: TF-IDF Pre-filter ---")
    s1b_candidates = tfidf_prefilter(s1_candidates, JD_PATH, top_k=5000)
    print(f"Stage 1.5 Filtering: Retained {len(s1b_candidates)} candidates.")
    
    # 2. STAGE 2: Semantic + Feature Extraction
    print("\n--- STAGE 2: Semantic Matching + Feature Extraction ---")
    matcher = EnsembleMatcher()
    matcher.load_jd(JD_PATH)
    s2_candidates = matcher.score_candidates(s1b_candidates)
    
    # Stage 3: Expanded scorecard ranking
    print("\n--- STAGE 3: 12-Feature Scorecard Ranking ---")
    top_candidates = rank_candidates(s2_candidates, top_n=100)
    print(f"Ranking complete. Selected top {len(top_candidates)} candidates.")
    
    # Stage 4: Recruiter-style explanations + debug output
    print("\n--- STAGE 4: Recruiter-Style Explanations & Output ---")
    output_filename = BASE_DIR / ("sample_submission_output.csv" if use_sample else "submission.csv")
    debug_filename = BASE_DIR / ("sample_debug_output.csv" if use_sample else "submission_debug.csv")
    
    generate_submission_csv(top_candidates, output_path=str(output_filename))
    generate_debug_csv(top_candidates, output_path=str(debug_filename))
    print_feature_diagnostics(top_candidates)
    
    top20_filename = BASE_DIR / ("sample_top20_debug.csv" if use_sample else "top20_debug.csv")
    generate_top20_debug_csv(top_candidates, output_path=str(top20_filename))
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n--- Pipeline Completed in {elapsed:.2f} seconds ---")

    # Print a quick summary of top 5
    print("\n--- Top 5 Candidates ---")
    for i, c in enumerate(top_candidates[:5], 1):
        sc = c.get("scorecard", {})
        print(f"  #{i}: {c.get('candidate_id', '?')} | "
              f"Score: {sc.get('overall_score', 0):.4f} | "
              f"Retrieval: {sc.get('retrieval_relevance', 0):.2f} | "
              f"Product: {sc.get('product_execution', 0):.2f} | "
              f"Title: {c.get('profile', {}).get('current_title', '?')}")

if __name__ == "__main__":
    use_sample = True
    if len(sys.argv) > 1 and sys.argv[1].lower() == "full":
        use_sample = False
    
    run_pipeline(use_sample=use_sample)
