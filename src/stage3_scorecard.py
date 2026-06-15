from src.config import WEIGHTS

def build_scorecard(cand):
    """
    Constructs an HRClaw-style scorecard object for the candidate,
    calculating the final behavioral score and the ensemble overall score.
    """
    semantic = cand.get("semantic_score", 0.0)
    hard_skill = cand.get("hard_skill_score", 0.0)
    
    signals = cand.get("redrob_signals", {})
    profile = cand.get("profile", {})
    
    # 1. Behavioral Score Calculation
    commits = min(10.0, signals.get("github_activity_score", 0.0)) / 10.0
    views = min(50, signals.get("profile_views_received_30d", 0)) / 50.0
    
    notice = signals.get("notice_period_days", 0)
    notice_score = max(0.0, (60.0 - notice) / 60.0)
    
    interview_rate = signals.get("interview_completion_rate", 0.5)
    offer_rate = signals.get("offer_acceptance_rate", 0.5)
    
    # Behavioral average with 5 conversion metrics
    behavioral = (commits + views + notice_score + interview_rate + offer_rate) / 5.0
    
    # Location score (JD Line 48/60): tagged in Stage 1
    location_score = cand.get("location_score", 0.5)
    
    # Experience sweet-spot bonus (JD Line 57: "6-8 years... 4-5 in applied ML/AI")
    rel_years = cand.get("relevant_years", 0.0)
    if 5.0 <= rel_years <= 9.0:
        exp_bonus = 1.0  # In the sweet spot
    elif 4.0 <= rel_years < 5.0 or 9.0 < rel_years <= 12.0:
        exp_bonus = 0.7  # Close to sweet spot
    else:
        exp_bonus = 0.4  # Outside sweet spot
    
    # 2. Final Ensemble Score (location and experience band are mild modifiers)
    w_sem = WEIGHTS["semantic"]
    w_hard = WEIGHTS["hard_skills"]
    w_beh = WEIGHTS["behavioral"]
    
    base_score = (semantic * w_sem) + (hard_skill * w_hard) + (behavioral * w_beh)
    # Apply modifiers: 80% base, 5% location, 15% experience sweet-spot
    overall = base_score * 0.80 + (location_score * 0.05) + (exp_bonus * 0.15)
    
    # 3. Compile Scorecard
    scorecard = {
        "overall_score": overall,
        "semantic_match": semantic,
        "hard_skill_match": hard_skill,
        "behavioral_match": behavioral,
        "evidence": {
            "years_exp": profile.get("years_of_experience", 0),
            "relevant_years": cand.get("relevant_years", 0.0),
            "notice_days": notice,
            "github_score": signals.get("github_activity_score", 0.0),
            "recruiter_views": signals.get("profile_views_received_30d", 0),
            "matched_skills": cand.get("matched_skills", [])
        }
    }
    
    cand["scorecard"] = scorecard
    cand["score"] = overall # Used for easy sorting
    return cand

def rank_candidates(candidates, top_n=100):
    """
    Evaluates all candidates via Scorecard and returns the top N ranked.
    """
    scored = [build_scorecard(c) for c in candidates]
    # Sort by: Score (Desc, rounded to 4 decimals), Candidate ID (Asc)
    # Note: validate_submission.py strictly requires candidate_id ascending for ties.
    scored.sort(key=lambda x: (
        -round(x["score"], 4),
        x.get("candidate_id", "")
    ))
    return scored[:top_n]
