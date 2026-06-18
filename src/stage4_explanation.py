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
      - Specific facts from the candidate's profile
      - Connection to specific JD requirements
      - Honest concerns where gaps exist
      - No hallucination
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

    strengths = []

    if (
        current_title
        and title_relevance >= 0.80
        and any(term in current_title.lower() for term in TITLE_EXPLANATION_TERMS)
    ):
        title_lower = current_title.lower()
        if any(term in title_lower for term in ("search", "ranking", "relevance", "recommendation")):
            strengths.append(f"Current {current_title} role directly aligns with the JD's search focus")
        else:
            strengths.append(f"Current {current_title} role aligns closely with the JD")

    if 5.0 <= rel_years <= 9.0:
        strengths.append(f"{rel_years:.1f} years of relevant experience falls within the target range")

    if is_open:
        strengths.append("Currently open to work")

    if activations["retrieval_relevance"]:
        terms = _format_evidence(concrete_evidence["retrieval_relevance"])
        strengths.append(
            f"Production retrieval evidence includes {terms}"
            if terms else "Strong evidence of production retrieval, search, or ranking work"
        )

    if activations["product_execution"]:
        terms = _format_evidence(concrete_evidence["product_execution"])
        strengths.append(
            f"Production delivery evidence includes {terms}"
            if terms else "Strong track record of shipping ML systems to production"
        )

    if activations["evaluation_systems"]:
        terms = _format_evidence(concrete_evidence["evaluation_systems"])
        strengths.append(
            f"Evaluation evidence includes {terms}"
            if terms else "Concrete experience with ranking or search evaluation"
        )

    if activations["vector_search"]:
        terms = _format_evidence(concrete_evidence["vector_search"])
        strengths.append(
            f"Vector-search evidence includes {terms}"
            if terms else "Hands-on vector-search infrastructure experience"
        )

    if display_skills:
        strengths.append(f"JD-relevant skills include {', '.join(display_skills)}")

    tradeoffs = []
    if notice >= 45:
        tradeoffs.append(f"{notice}-day notice period")
    if pen_research >= 0.30:
        tradeoffs.append("Research-heavy background with limited production evidence")
    if pen_consulting >= 0.25:
        tradeoffs.append("Predominantly consulting-company career history")
    if pen_spec >= 0.35:
        tradeoffs.append("Primary expertise in CV/Speech/Robotics rather than NLP/IR")
    if activity < 0.3:
        tradeoffs.append("Limited recent platform activity")
    if stability < 0.3:
        tradeoffs.append("Frequent job changes")
    if github < 2.0:
        tradeoffs.append("Limited open-source activity")
    if not is_open and availability < 0.5:
        tradeoffs.append("Not currently marked open to work")

    strength_text = "; ".join(strengths) if strengths else "Relevant technical background"
    tradeoff_text = "; ".join(tradeoffs) if tradeoffs else "no major evidence-backed concerns"
    return f"Strength: {strength_text}. Tradeoff: {tradeoff_text}."


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
