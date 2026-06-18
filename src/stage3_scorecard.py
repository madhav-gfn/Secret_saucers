from src.config import FEATURE_WEIGHTS, PENALTY_CAPS, HARD_SKILL_BLEND

def build_scorecard(cand):
    """
    Constructs the final scorecard using the expanded 12-feature weighting system.
    
    Design rationale:
      - Semantic embedding (18%) provides broad retrieval but doesn't dominate.
      - Retrieval relevance (15%) is the top engineered feature — directly targets
        the JD's explicit requirement for production retrieval/ranking/search people.
      - Product execution (12%) separates shippers from researchers.
      - Evaluation systems (10%) rewards candidates who've built eval frameworks.
      - Penalties are applied as small deductions, not binary filters.
    """
    # ── Retrieve all pre-computed feature scores ──
    semantic = cand.get("semantic_score", 0.0)
    hard_skill = cand.get("hard_skill_score", 0.0)

    # Blend semantic + hard_skill into the "semantic" slot
    # This gives hard-skill matching some weight within the embedding signal
    blended_semantic = (semantic * (1 - HARD_SKILL_BLEND)) + (hard_skill * HARD_SKILL_BLEND)

    # Domain-specific features from features.py
    feat_retrieval   = cand.get("feat_retrieval_relevance", 0.0)
    feat_product     = cand.get("feat_product_execution", 0.0)
    feat_product_comp= cand.get("feat_product_company", 0.0)
    feat_eval        = cand.get("feat_evaluation_systems", 0.0)
    feat_title       = cand.get("feat_title_relevance", 0.0)
    feat_exp_align   = cand.get("feat_experience_alignment", 0.0)
    feat_stability   = cand.get("feat_career_stability", 0.0)
    feat_activity    = cand.get("feat_activity", 0.0)
    feat_avail       = cand.get("feat_availability", 0.0)
    feat_vector      = cand.get("feat_vector_search", 0.0)
    feat_nlp         = cand.get("feat_nlp_llm", 0.0)
    feat_consistency = cand.get("feat_consistency", 0.0)

    # Penalties
    pen_research     = cand.get("pen_research", 0.0)
    pen_spec         = cand.get("pen_specialization", 0.0)
    pen_consulting   = cand.get("pen_consulting", 0.0)

    # Location score (from Stage 1 tagging)
    location_score   = cand.get("location_score", 0.5)

    # ── Weighted sum ──
    w = FEATURE_WEIGHTS
    base_score = (
        blended_semantic     * w["semantic"]
        + feat_retrieval     * w["retrieval_relevance"]
        + feat_product       * w["product_execution"]
        + feat_product_comp  * w.get("product_company", 0.0)
        + feat_eval          * w["evaluation_systems"]
        + feat_title         * w["title_relevance"]
        + feat_exp_align     * w["experience_alignment"]
        + feat_stability     * w["career_stability"]
        + feat_activity      * w["activity"]
        + feat_avail         * w["availability"]
        + feat_vector        * w["vector_search"]
        + feat_nlp           * w["nlp_llm"]
        + feat_consistency   * w["consistency"]
    )

    # ── Apply penalties as deductions (capped) ──
    p = PENALTY_CAPS
    penalty_total = (
        pen_research    * p["research"]
        + pen_spec      * p["specialization"]
        + pen_consulting * p["consulting"]
    )

    # Location is a small bonus (not in the main weight table)
    location_bonus = location_score * 0.02

    overall = max(0.0, min(1.0, base_score - penalty_total + location_bonus))

    # ── Compile scorecard for explanation generation ──
    signals = cand.get("redrob_signals", {})
    profile = cand.get("profile", {})

    scorecard = {
        "overall_score": overall,
        # Core scores
        "semantic_match": semantic,
        "hard_skill_match": hard_skill,
        "blended_semantic": blended_semantic,
        # Domain features
        "retrieval_relevance": feat_retrieval,
        "product_execution": feat_product,
        "product_company": feat_product_comp,
        "evaluation_systems": feat_eval,
        "vector_search": feat_vector,
        "nlp_llm": feat_nlp,
        "title_relevance": feat_title,
        "experience_alignment": feat_exp_align,
        "career_stability": feat_stability,
        "activity": feat_activity,
        "availability": feat_avail,
        "consistency": feat_consistency,
        # Penalties
        "research_penalty": pen_research,
        "specialization_penalty": pen_spec,
        "consulting_penalty": pen_consulting,
        "penalty_total": penalty_total,
        # Evidence for explanations
        "evidence": {
            "years_exp": profile.get("years_of_experience", 0),
            "relevant_years": cand.get("relevant_years", 0.0),
            "notice_days": signals.get("notice_period_days", 0),
            "github_score": signals.get("github_activity_score", 0.0),
            "recruiter_views": signals.get("profile_views_received_30d", 0),
            "open_to_work": signals.get("open_to_work_flag", False),
            "matched_skills": cand.get("matched_skills", []),
            "current_title": profile.get("current_title", ""),
            "location": profile.get("location", ""),
        }
    }
    
    cand["scorecard"] = scorecard
    cand["score"] = overall
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
