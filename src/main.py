import sys
import time
from src.config import SAMPLE_CANDIDATES_PATH, FULL_CANDIDATES_PATH, JD_PATH, BASE_DIR
from src.stage1_filter import stream_and_filter_candidates
from src.stage2_ensemble import EnsembleMatcher
from src.stage3_scorecard import rank_candidates
from src.stage4_explanation import generate_submission_csv
from src.jd_analyzer import extract_dynamic_skills

def run_pipeline(use_sample=False):
    data_path = SAMPLE_CANDIDATES_PATH if use_sample else FULL_CANDIDATES_PATH
    output_path = BASE_DIR / "submission.csv"
    
    start_time = time.time()
    
    # 0. DYNAMIC LLM JD PARSING
    # Attempt to extract exact requirements vs rejected skills
    with open(JD_PATH, "r", encoding="utf-8") as f:
        jd_text = f.read()
    dynamic_req, dynamic_rej = extract_dynamic_skills(jd_text)
    
    print(f"--- Starting Synthesized Pipeline with data: {data_path.name} ---")
    
    # 1. STAGE 1: Viability Pruning
    print("\n--- STAGE 1: Fast Pruning (Honeypot & Baseline) ---")
    s1_candidates = list(stream_and_filter_candidates(data_path))
    print(f"Stage 1 Filtering: Retained {len(s1_candidates)} candidates.")
    
    # 2. STAGE 2: Deep Semantic & Hard Skill Matching
    print("\n--- STAGE 2: Feature Extraction & Ensemble Matching ---")
    matcher = EnsembleMatcher()
    matcher.load_jd(JD_PATH, dynamic_req, dynamic_rej)
    s2_candidates = matcher.score_candidates(s1_candidates)
    
    # Stage 3
    print("\n--- STAGE 3: Scorecard Evaluation & Ranking ---")
    top_candidates = rank_candidates(s2_candidates, top_n=100)
    print(f"Ranking complete. Selected top {len(top_candidates)} candidates.")
    
    # Stage 4
    print("\n--- STAGE 4: Transparent Explanations & Output ---")
    output_filename = BASE_DIR / ("sample_submission_output.csv" if use_sample else "submission.csv")
    generate_submission_csv(top_candidates, output_path=str(output_filename))
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n--- Pipeline Completed in {elapsed:.2f} seconds ---")

if __name__ == "__main__":
    use_sample = True
    if len(sys.argv) > 1 and sys.argv[1].lower() == "full":
        use_sample = False
    
    run_pipeline(use_sample=use_sample)
