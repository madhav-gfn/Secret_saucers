import pandas as pd

# Technical skill categories for display filtering
# Only show these categories in explanations to avoid noise like "Marketing", "Content Writing"
TECHNICAL_CATEGORIES = {
    # Core AI/ML
    "python", "pytorch", "tensorflow", "keras", "scikit", "sklearn", "numpy", "pandas",
    "deep learning", "machine learning", "neural", "nlp", "natural language",
    "llm", "gpt", "bert", "transformer", "attention", "fine-tuning", "lora", "qlora", "peft",
    "rag", "langchain", "llamaindex", "hugging face", "embeddings", "sentence-transformers",
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

def generate_reasoning(cand):
    """
    Generates a transparent, evidence-backed 1-2 sentence explanation 
    using the HRClaw Scorecard, bypassing the need for a slow LLM.
    """
    sc = cand.get("scorecard", {})
    ev = sc.get("evidence", {})
    
    score = sc.get("overall_score", 0.0)
    semantic = sc.get("semantic_match", 0.0)
    hard = sc.get("hard_skill_match", 0.0)
    beh = sc.get("behavioral_match", 0.0)
    rel_years = ev.get("relevant_years", 0.0)
    
    # 1. Opening Rating (varied language)
    if score >= 0.8:
        rating = "Excellent match"
    elif score >= 0.7:
        rating = "Strong match"
    elif score >= 0.6:
        rating = "Good match"
    else:
        rating = "Potential match"
        
    reasoning = f"{rating} (Overall: {score*100:.1f}%). "
    
    # 2. Filter matched skills to only show technical ones
    raw_matched = ev.get("matched_skills", [])
    tech_matched = [s for s in raw_matched if is_technical_skill(s)]
    if not tech_matched and raw_matched:
        tech_matched = raw_matched[:2]  # Fallback: show first 2 if no tech match
    
    if len(tech_matched) > 3:
        display_skills = tech_matched[:3]
        skills_str = ", ".join(display_skills) + ", etc"
    elif tech_matched:
        skills_str = ", ".join(tech_matched)
    else:
        skills_str = ""

    # 3. Highlight strongest signal with varied language
    if semantic >= hard and semantic >= beh:
        if semantic > 0.7:
            reasoning += f"Career history and profile strongly align with the JD's core requirements ({semantic*100:.1f}% semantic match). "
        else:
            reasoning += f"Good semantic alignment with the role's AI/ML focus ({semantic*100:.1f}%). "
    elif hard >= semantic and hard >= beh:
        if skills_str:
            reasoning += f"Demonstrates key technical competencies: {skills_str} ({hard*100:.1f}% skill match). "
        else:
            reasoning += f"Strong technical skill alignment ({hard*100:.1f}%). "
    else:
        if beh > 0.85:
            reasoning += f"Excellent availability and engagement signals ({beh*100:.1f}% behavioral score). "
        else:
            reasoning += f"Favorable hiring readiness indicators ({beh*100:.1f}%). "
        
    # 4. Evidence line with experience context
    years_exp = ev.get('years_exp', 0)
    notice = ev.get('notice_days', 0)
    github = ev.get('github_score', 0)
    
    # Experience context relative to JD sweet spot
    if 5 <= rel_years <= 9:
        exp_context = f"{rel_years:.1f} yrs relevant experience (in JD sweet spot)"
    elif rel_years > 9:
        exp_context = f"{rel_years:.1f} yrs relevant experience (senior)"
    else:
        exp_context = f"{rel_years:.1f} yrs relevant experience"
    
    reasoning += f"Evidence: {exp_context}"
    
    if notice == 0:
        reasoning += ", immediate joiner"
    elif notice <= 30:
        reasoning += f", {notice} day notice"
    
    if github > 7.0:
        reasoning += f", active GitHub contributor (score: {github})."
    else:
        reasoning += "."
        
    return reasoning

def generate_submission_csv(candidates, output_path):
    """
    Takes the final ranked candidates, generates their reasoning, 
    and writes the submission.csv matching the expected format.
    """
    results = []
    
    for rank, cand in enumerate(candidates, start=1):
        reasoning = generate_reasoning(cand)
        results.append({
            "candidate_id": cand.get("candidate_id", ""),
            "rank": rank,
            "score": f"{cand.get('score', 0.0):.4f}",
            "reasoning": reasoning
        })
        
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Successfully wrote submission to {output_path}")
