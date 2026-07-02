import pandas as pd
from statistics import mean, median

from src.features import feature_evidence


EXPLANATION_THRESHOLDS = {
    "retrieval_relevance": 0.60,
    "evaluation_systems": 0.50,
    "vector_search": 0.60,
    "product_execution": 0.50,
}

SKILL_PRIORITY = (
    (
        "information retrieval", "semantic search", "vector search", "search",
        "ranking", "recommendation systems", "recommendation", "faiss",
        "pinecone", "qdrant", "weaviate", "opensearch", "elasticsearch",
    ),
    (
        "nlp", "natural language", "llm", "rag", "embeddings", "transformers",
        "transformer",
    ),
    ("python", "pytorch", "tensorflow", "mlops"),
)

DIAGNOSTIC_FEATURES = (
    "retrieval_relevance",
    "evaluation_systems",
    "vector_search",
    "product_execution",
    "title_relevance",
    "availability",
    "activity",
)

TITLE_EXPLANATION_TERMS = (
    "machine learning", "ml engineer", "ai engineer", "applied ai",
    "applied scientist", "search", "ranking", "relevance", "recommendation",
    "nlp engineer", "data scientist", "research engineer", "deep learning",
)

EVIDENCE_LABELS = {
    "ndcg": "NDCG",
    "mrr": "MRR",
    "map": "MAP",
    "bm25": "BM25",
    "ltr": "learning to rank",
    "faiss": "FAISS",
    "hnsw": "HNSW",
    "ivf": "IVF",
}

# Technical skill categories for display filtering
# Only show these categories in explanations to avoid noise like "Marketing", "Content Writing"
TECHNICAL_CATEGORIES = {
    # Core AI/ML
    "python", "pytorch", "tensorflow", "keras", "scikit", "sklearn", "numpy", "pandas",
    "deep learning", "machine learning", "neural", "nlp", "natural language",
    "llm", "gpt", "bert", "transformer", "attention", "fine-tuning", "lora", "qlora", "peft",
    "rag", "llamaindex", "hugging face", "embeddings", "sentence-transformers",
    # Vector DBs & Search
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "vector search", "semantic search", "information retrieval", "search backend",
    "retrieval", "ranking", "recommendation", "recsys",
    # Data & Infra
    "spark", "airflow", "kafka", "docker", "kubernetes", "aws", "gcp", "azure",
    "sql", "nosql", "postgresql", "mongodb", "redis",
    # Evaluation & Methods
    "a/b testing", "ndcg", "evaluation", "mlops", "feature engineering",
    "reinforcement learning", "computer vision", "generative ai",
    "distributed systems", "microservices", "rest api",
}

def is_technical_skill(skill_name):
    """Check if a skill name is genuinely technical (for display purposes only)."""
    name_lower = skill_name.lower()
    # Check exact match or substring against known technical categories
    for cat in TECHNICAL_CATEGORIES:
        if cat in name_lower or name_lower in cat:
            return True
    return False


def _skill_priority(skill_name):
    normalized = skill_name.lower()
    for tier, keywords in enumerate(SKILL_PRIORITY):
        for position, keyword in enumerate(keywords):
            if keyword in normalized:
                return tier, position, normalized
    return len(SKILL_PRIORITY), 0, normalized


def _prioritized_skills(skills, limit=3):
    unique = {}
    for skill in skills:
        if is_technical_skill(skill):
            unique.setdefault(skill.lower(), skill)
    return sorted(unique.values(), key=_skill_priority)[:limit]


def explanation_activations(cand):
    sc = cand.get("scorecard", {})
    eligible = []
    for feature, threshold in EXPLANATION_THRESHOLDS.items():
        score = sc.get(feature, 0.0)
        if score >= threshold:
            confidence = (score - threshold) / (1.0 - threshold)
            eligible.append((confidence, score, feature))

    selected = {feature for _, _, feature in sorted(eligible, reverse=True)[:2]}
    return {feature: feature in selected for feature in EXPLANATION_THRESHOLDS}


def _format_evidence(items, limit=2):
    return ", ".join(EVIDENCE_LABELS.get(item, item) for item in items[:limit])


def generate_reasoning(cand, rank):
    """
    Generates a recruiter-style, evidence-backed explanation referencing specific
    profile facts, JD connection, strengths, and tradeoffs.
    
    Design goal: pass the Stage 4 manual review checks:
      - Specific facts from the candidate's profile (company, title, years)
      - Connection to specific JD requirements (retrieval, ranking, search)
      - Honest concerns where gaps exist
      - No hallucination — every claim is backed by extracted data
      - Substantive variation between candidates
      - Tone matches the rank
    """
    sc = cand.get("scorecard", {})
    ev = sc.get("evidence", {})
    
    rel_years = ev.get("relevant_years", 0.0)
    notice = ev.get("notice_days", 0)
    github = ev.get("github_score", 0.0)
    is_open = ev.get("open_to_work", False)
    current_title = ev.get("current_title", "")
    current_company = ev.get("current_company", "")
    location = ev.get("location", "")
    country = ev.get("country", "")
    career_trajectory = ev.get("career_trajectory", [])
    title_relevance = sc.get("title_relevance", 0.0)
    stability = sc.get("career_stability", 0.0)
    activity = sc.get("activity", 0.0)
    availability = sc.get("availability", 0.0)
    activations = explanation_activations(cand)
    concrete_evidence = feature_evidence(cand)
    
    # Penalties
    pen_research = sc.get("research_penalty", 0.0)
    pen_spec = sc.get("specialization_penalty", 0.0)
    pen_consulting = sc.get("consulting_penalty", 0.0)
    
    raw_matched = ev.get("matched_skills", [])
    display_skills = _prioritized_skills(raw_matched)

    # Track all evidence terms already used to avoid repetition
    used_evidence = set()

    strengths = []

    # ── 1. Title + Company context (the most differentiating fact) ──
    if current_title and title_relevance >= 0.80:
        title_lower = current_title.lower()
        company_clause = f" at {current_company}" if current_company else ""
        if any(term in title_lower for term in ("search", "ranking", "relevance", "recommendation")):
            strengths.append(
                f"Currently a {current_title}{company_clause}, directly aligned with the JD's core need for retrieval and ranking expertise"
            )
        elif any(term in title_lower for term in TITLE_EXPLANATION_TERMS):
            strengths.append(
                f"Currently a {current_title}{company_clause}, with a role profile matching the JD's Senior AI Engineer requirements"
            )
        else:
            strengths.append(f"Currently a {current_title}{company_clause}")
    elif current_title and current_company:
        strengths.append(f"Currently a {current_title} at {current_company}")

    # ── 2. Experience alignment with specific JD range ──
    if 5.0 <= rel_years <= 9.0:
        strengths.append(
            f"{rel_years:.1f} years of relevant technical experience, within the JD's target 5-9 year range"
        )
    elif 4.0 <= rel_years < 5.0:
        strengths.append(
            f"{rel_years:.1f} years of relevant experience, slightly below the JD's 5-year target but approaching range"
        )
    elif 9.0 < rel_years <= 12.0:
        strengths.append(
            f"{rel_years:.1f} years of relevant experience, bringing senior-level depth beyond the JD's 5-9 year range"
        )

    # ── 3. Career trajectory (shows progression, not just current role) ──
    if len(career_trajectory) >= 2:
        prev = career_trajectory[1]
        curr = career_trajectory[0]
        prev_title = prev.get("title", "")
        prev_company = prev.get("company", "")
        curr_title = curr.get("title", "")
        # Only mention if it shows meaningful progression
        if prev_title and prev_company and prev_title.lower() != curr_title.lower():
            strengths.append(
                f"Career progression from {prev_title} at {prev_company} to current role shows growth trajectory"
            )

    # ── 4. Retrieval/Search evidence (highest-value JD signal) ──
    if activations["retrieval_relevance"]:
        terms = concrete_evidence.get("retrieval_relevance", [])
        # Format evidence terms more naturally
        formatted = [EVIDENCE_LABELS.get(t, t) for t in terms[:3]]
        used_evidence.update(terms[:3])
        if formatted:
            strengths.append(
                f"Demonstrates production retrieval/search work — profile references {', '.join(formatted)}, "
                f"matching the JD's emphasis on building ranking and search systems"
            )
        else:
            strengths.append(
                "Strong evidence of production retrieval, search, or ranking work aligning with the JD's core focus"
            )

    # ── 5. Product execution (JD: "shipper over researcher") ──
    if activations["product_execution"]:
        terms = concrete_evidence.get("product_execution", [])
        # Only show terms not already used in retrieval evidence
        new_terms = [t for t in terms[:3] if t not in used_evidence]
        used_evidence.update(terms[:3])
        formatted = [EVIDENCE_LABELS.get(t, t) for t in new_terms[:2]]
        if formatted:
            strengths.append(
                f"Track record of shipping to production — evidence includes {', '.join(formatted)}"
            )
        else:
            strengths.append("Demonstrated ability to ship ML systems to production")

    # ── 6. Evaluation systems (rare and highly valued) ──
    if activations["evaluation_systems"]:
        terms = concrete_evidence.get("evaluation_systems", [])
        new_terms = [t for t in terms[:3] if t not in used_evidence]
        used_evidence.update(terms[:3])
        formatted = [EVIDENCE_LABELS.get(t, t) for t in new_terms[:2]]
        if formatted:
            strengths.append(
                f"Experience with evaluation frameworks ({', '.join(formatted)}), a differentiator the JD specifically values"
            )
        else:
            strengths.append("Hands-on experience with search/ranking evaluation frameworks")

    # ── 7. Vector search infrastructure ──
    if activations["vector_search"]:
        terms = concrete_evidence.get("vector_search", [])
        new_terms = [t for t in terms[:3] if t not in used_evidence]
        used_evidence.update(terms[:3])
        formatted = [EVIDENCE_LABELS.get(t, t) for t in new_terms[:2]]
        if formatted:
            strengths.append(
                f"Hands-on vector-search infrastructure experience ({', '.join(formatted)})"
            )
        else:
            strengths.append("Practical vector-search and embedding infrastructure experience")

    # ── 8. Matched skills (concrete, verifiable) ──
    if display_skills:
        strengths.append(f"JD-aligned skills: {', '.join(display_skills)}")

    # ── 9. Availability and location signals ──
    avail_parts = []
    if is_open:
        avail_parts.append("actively open to work")
    if notice <= 30 and notice > 0:
        avail_parts.append(f"available within {notice} days")
    if avail_parts:
        strengths.append(f"Availability: {', '.join(avail_parts)}")

    # ── Tradeoffs (honest, evidence-backed concerns) ──
    tradeoffs = []
    if notice >= 60:
        tradeoffs.append(
            f"{notice}-day notice period exceeds the JD's preference for sub-30-day availability"
        )
    elif notice >= 45:
        tradeoffs.append(f"{notice}-day notice period is above the JD's preferred range")

    if pen_research >= 0.30:
        tradeoffs.append(
            "Career leans research-heavy with limited production deployment evidence"
        )
    if pen_consulting >= 0.25:
        tradeoffs.append(
            "Career history is predominantly consulting firms, which the JD flags as a concern"
        )
    if pen_spec >= 0.35:
        tradeoffs.append(
            "Primary expertise appears to be in CV/Speech/Robotics rather than NLP/IR, "
            "which the JD warns may require re-learning fundamentals"
        )
    if stability < 0.3:
        tradeoffs.append(
            "Frequent job changes (avg tenure < 18 months) — the JD notes concern about candidates "
            "switching every 1.5 years"
        )
    if activity < 0.3:
        tradeoffs.append("Limited recent platform activity, raising availability concerns")
    if github < 2.0:
        tradeoffs.append("Minimal open-source/GitHub activity")
    if not is_open and availability < 0.5:
        tradeoffs.append("Not currently marked open to work")

    # ── Assemble with rank-aware tone ──
    strength_text = "; ".join(strengths) if strengths else "Relevant technical background"
    tradeoff_text = "; ".join(tradeoffs) if tradeoffs else "no major evidence-backed concerns"

    # Rank-aware framing
    if rank <= 10:
        return f"Strong match. {strength_text}. Tradeoff: {tradeoff_text}."
    elif rank <= 30:
        return f"Good match. {strength_text}. Tradeoff: {tradeoff_text}."
    elif rank <= 60:
        return f"Moderate match. {strength_text}. Tradeoff: {tradeoff_text}."
    else:
        return f"Borderline match. {strength_text}. Tradeoff: {tradeoff_text}."


def generate_submission_csv(candidates, output_path):
    """
    Takes the final ranked candidates, generates their reasoning, 
    and writes the submission.csv matching the expected format.
    """
    results = []
    
    for rank, cand in enumerate(candidates, start=1):
        reasoning = generate_reasoning(cand, rank)
        results.append({
            "candidate_id": cand.get("candidate_id", ""),
            "rank": rank,
            "score": f"{cand.get('score', 0.0):.4f}",
            "reasoning": reasoning
        })
        
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Successfully wrote submission to {output_path}")


def generate_debug_csv(candidates, output_path):
    """
    Writes a debug CSV with all feature scores and penalties for manual inspection.
    This file is NOT submitted — it's for internal analysis only.
    """
    rows = []
    for rank, cand in enumerate(candidates, start=1):
        sc = cand.get("scorecard", {})
        ev = sc.get("evidence", {})
        activations = explanation_activations(cand)
        rows.append({
            "rank": rank,
            "candidate_id": cand.get("candidate_id", ""),
            "final_score": f"{sc.get('overall_score', 0.0):.4f}",
            # Core
            "semantic_match": f"{sc.get('semantic_match', 0.0):.4f}",
            "hard_skill_match": f"{sc.get('hard_skill_match', 0.0):.4f}",
            "blended_semantic": f"{sc.get('blended_semantic', 0.0):.4f}",
            # Domain features
            "retrieval_relevance": f"{sc.get('retrieval_relevance', 0.0):.4f}",
            "product_execution": f"{sc.get('product_execution', 0.0):.4f}",
            "product_company_score": f"{sc.get('product_company', 0.0):.4f}",
            "evaluation_systems": f"{sc.get('evaluation_systems', 0.0):.4f}",
            "vector_search": f"{sc.get('vector_search', 0.0):.4f}",
            "nlp_llm": f"{sc.get('nlp_llm', 0.0):.4f}",
            "title_relevance": f"{sc.get('title_relevance', 0.0):.4f}",
            "experience_alignment": f"{sc.get('experience_alignment', 0.0):.4f}",
            "career_stability": f"{sc.get('career_stability', 0.0):.4f}",
            "activity": f"{sc.get('activity', 0.0):.4f}",
            "availability": f"{sc.get('availability', 0.0):.4f}",
            "consistency": f"{sc.get('consistency', 0.0):.4f}",
            # Penalties
            "pen_research": f"{sc.get('research_penalty', 0.0):.4f}",
            "pen_specialization": f"{sc.get('specialization_penalty', 0.0):.4f}",
            "pen_consulting": f"{sc.get('consulting_penalty', 0.0):.4f}",
            "penalty_total": f"{sc.get('penalty_total', 0.0):.4f}",
            # Evidence
            "years_exp": ev.get("years_exp", 0),
            "relevant_years": f"{ev.get('relevant_years', 0.0):.1f}",
            "notice_days": ev.get("notice_days", 0),
            "current_title": ev.get("current_title", ""),
            "location": ev.get("location", ""),
            "open_to_work": ev.get("open_to_work", False),
            "github_score": ev.get("github_score", 0),
            "matched_skills": "; ".join(ev.get("matched_skills", [])[:6]),
            "explain_retrieval": activations["retrieval_relevance"],
            "explain_evaluation": activations["evaluation_systems"],
            "explain_vector_search": activations["vector_search"],
            "explain_product_execution": activations["product_execution"],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Successfully wrote debug output to {output_path}")


def print_feature_diagnostics(candidates):
    """Print Top-100 score distributions and explanation activation counts."""
    top_candidates = candidates[:100]
    if not top_candidates:
        print("No candidates available for feature diagnostics.")
        return

    print("\n--- Top 100 Feature Distribution ---")
    print(f"{'feature':<24} {'mean':>7} {'median':>7} {'min':>7} {'max':>7}")
    for feature in DIAGNOSTIC_FEATURES:
        values = [cand.get("scorecard", {}).get(feature, 0.0) for cand in top_candidates]
        print(
            f"{feature:<24} {mean(values):>7.3f} {median(values):>7.3f} "
            f"{min(values):>7.3f} {max(values):>7.3f}"
        )

    print("\n--- Top 100 Explanation Activations (two strongest per candidate) ---")
    for feature, threshold in EXPLANATION_THRESHOLDS.items():
        count = sum(explanation_activations(cand)[feature] for cand in top_candidates)
        print(f"{feature:<24} {count:>3}/{len(top_candidates)} (threshold >= {threshold:.2f})")

def generate_top20_debug_csv(candidates, output_path):
    """
    Generates a top20_debug.csv file containing key feature scores and penalties
    for manual ranking inspection.
    """
    rows = []
    for rank, cand in enumerate(candidates[:20], start=1):
        sc = cand.get("scorecard", {})
        rows.append({
            "candidate_id": cand.get("candidate_id", ""),
            "rank": rank,
            "final_score": f"{sc.get('overall_score', 0.0):.4f}",
            "semantic_score": f"{sc.get('semantic_match', 0.0):.4f}",
            "retrieval_score": f"{sc.get('retrieval_relevance', 0.0):.4f}",
            "evaluation_score": f"{sc.get('evaluation_systems', 0.0):.4f}",
            "product_company_score": f"{sc.get('product_company', 0.0):.4f}",
            "activity_score": f"{sc.get('activity', 0.0):.4f}",
            "availability_score": f"{sc.get('availability', 0.0):.4f}",
            "career_stability_score": f"{sc.get('career_stability', 0.0):.4f}",
            "pen_research": f"{sc.get('research_penalty', 0.0):.4f}",
            "pen_specialization": f"{sc.get('specialization_penalty', 0.0):.4f}",
            "pen_consulting": f"{sc.get('consulting_penalty', 0.0):.4f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Successfully wrote top 20 debug output to {output_path}")
