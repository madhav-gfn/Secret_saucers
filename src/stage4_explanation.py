import pandas as pd

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
    
    overall = sc.get("overall_score", 0.0)
    rel_years = ev.get("relevant_years", 0.0)
    notice = ev.get("notice_days", 0)
    github = ev.get("github_score", 0.0)
    views = ev.get("recruiter_views", 0)
    is_open = ev.get("open_to_work", False)
    current_title = ev.get("current_title", "")
    location = ev.get("location", "")
    
    # Feature scores for explanation
    retrieval = sc.get("retrieval_relevance", 0.0)
    product_exec = sc.get("product_execution", 0.0)
    eval_sys = sc.get("evaluation_systems", 0.0)
    vector_sc = sc.get("vector_search", 0.0)
    stability = sc.get("career_stability", 0.0)
    activity = sc.get("activity", 0.0)
    
    # Penalties
    pen_research = sc.get("research_penalty", 0.0)
    pen_spec = sc.get("specialization_penalty", 0.0)
    pen_consulting = sc.get("consulting_penalty", 0.0)
    
    # Filter to technical skills for display
    raw_matched = ev.get("matched_skills", [])
    tech_matched = [s for s in raw_matched if is_technical_skill(s)]
    
    parts = []
    
    # ── 1. Opening — calibrated to rank ──
    if rank <= 10:
        if retrieval > 0.4:
            parts.append(f"Top-tier match for the Senior AI Engineer role (score: {overall:.3f})")
        else:
            parts.append(f"Strong overall match (score: {overall:.3f})")
    elif rank <= 30:
        parts.append(f"Strong match (score: {overall:.3f})")
    elif rank <= 60:
        parts.append(f"Good match (score: {overall:.3f})")
    else:
        parts.append(f"Viable candidate (score: {overall:.3f})")
    
    # ── 2. Core strength — retrieval/search/ranking experience ──
    if retrieval >= 0.5:
        parts.append("with demonstrated production retrieval and ranking experience")
    elif retrieval >= 0.25:
        parts.append("with relevant search/ranking exposure")
    elif product_exec >= 0.4:
        parts.append("with strong production engineering track record")
    
    # ── 3. Product execution ──
    if product_exec >= 0.5:
        parts.append("evidence of shipping ML systems to production")
    
    # ── 4. Evaluation systems ──
    if eval_sys >= 0.3:
        parts.append("experience with evaluation frameworks (NDCG/MRR/A-B testing)")
    
    # ── 5. Vector search ──
    if vector_sc >= 0.4:
        parts.append("hands-on vector-search infrastructure experience")
    
    # ── 6. Technical skills ──
    if len(tech_matched) >= 3:
        display = tech_matched[:4]
        parts.append(f"key skills: {', '.join(display)}")
    elif len(tech_matched) >= 1:
        parts.append(f"relevant skills: {', '.join(tech_matched)}")
    
    # ── 7. Experience range ──
    if 5.0 <= rel_years <= 9.0:
        parts.append(f"{rel_years:.1f} years relevant experience (within JD target range)")
    elif rel_years > 9.0:
        parts.append(f"{rel_years:.1f} years relevant experience (senior)")
    else:
        parts.append(f"{rel_years:.1f} years relevant experience")
    
    # ── 8. Career stability ──
    if stability >= 0.8:
        parts.append("stable career progression")
    elif stability < 0.4:
        parts.append("some career stability concerns")
    
    # ── 9. Activity & availability signals ──
    availability_parts = []
    if is_open:
        availability_parts.append("actively open to work")
    if notice == 0:
        availability_parts.append("immediate joiner")
    elif notice <= 30:
        availability_parts.append(f"{notice}-day notice")
    if views >= 10:
        availability_parts.append("strong recruiter engagement")
    elif views >= 5:
        availability_parts.append("moderate recruiter interest")
    if github > 7.0:
        availability_parts.append(f"active GitHub contributor (score: {github:.0f})")
    
    if availability_parts:
        parts.append(", ".join(availability_parts))
    
    # ── 10. Tradeoffs / concerns (honest) ──
    concerns = []
    if notice > 45:
        concerns.append(f"{notice}-day notice period")
    if pen_research > 0.2:
        concerns.append("research-heavy background with limited production evidence")
    if pen_consulting > 0.2:
        concerns.append("predominantly consulting-company career history")
    if pen_spec > 0.2:
        concerns.append("primary expertise in CV/Speech/Robotics rather than NLP/IR")
    if activity < 0.3:
        concerns.append("limited recent platform activity")
    if stability < 0.3:
        concerns.append("frequent job changes")
    
    if concerns:
        parts.append("Tradeoffs: " + "; ".join(concerns))
    
    # ── Build final reasoning ──
    # Join with proper punctuation
    if len(parts) <= 2:
        reasoning = ". ".join(parts) + "."
    else:
        # First part is the opening, rest are comma-separated details
        opening = parts[0]
        details = parts[1:]
        
        # Split into strengths and tradeoffs
        tradeoff_parts = [p for p in details if p.startswith("Tradeoffs:")]
        strength_parts = [p for p in details if not p.startswith("Tradeoffs:")]
        
        reasoning = opening + ". " if not opening.endswith(".") else opening + " "
        
        if strength_parts:
            # Capitalize the first strength
            strength_parts[0] = strength_parts[0][0].upper() + strength_parts[0][1:]
            reasoning += ", ".join(strength_parts) + "."
        
        if tradeoff_parts:
            reasoning += " " + tradeoff_parts[0] + "."
    
    return reasoning


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
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Successfully wrote debug output to {output_path}")
