"""
features.py — Domain-specific feature extraction for the Redrob Senior AI Engineer JD.

Every function accepts a candidate dict and returns a normalized float.
Scores: 0.0 (worst) to 1.0 (best).
Penalties: 0.0 (no penalty) to 1.0 (full penalty).
"""

import re
from datetime import datetime, timedelta


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────

def _gather_text(cand):
    """Collect all searchable text from a candidate into a single lowercase blob."""
    profile = cand.get("profile", {})
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]
    for job in cand.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for s in cand.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(parts).lower()


def _keyword_matches(text, keywords):
    """Find non-overlapping keyword matches using word-safe comparisons."""
    matches = []
    for kw in keywords:
        if " " in kw or "-" in kw:
            if kw in text:
                matches.append(kw)
        else:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                matches.append(kw)

    deduplicated = []
    for kw in sorted(matches, key=len, reverse=True):
        if not any(kw in longer for longer in deduplicated):
            deduplicated.append(kw)
    return sorted(deduplicated)


def _count_keyword_hits(text, keywords):
    """Count how many distinct, non-overlapping keywords appear in text."""
    return len(_keyword_matches(text, keywords))


def _career_text(cand):
    """Concatenate all career description text."""
    parts = []
    for job in cand.get("career_history", []):
        parts.append(job.get("description", ""))
        parts.append(job.get("title", ""))
    return " ".join(parts).lower()


def _matched_keywords(text, keywords):
    """Return the distinct keywords present in text using the scoring matcher."""
    return _keyword_matches(text, keywords)


# ──────────────────────────────────────────────
# A. RETRIEVAL / SEARCH RELEVANCE  (highest impact)
# ──────────────────────────────────────────────

_RETRIEVAL_EXPLICIT = {
    "information retrieval", "search engine", "search system",
    "search quality", "search relevance", "search backend",
    "ranking system", "ranking model", "learning to rank", "learning-to-rank",
    "candidate matching", "relevance engineering",
    "marketplace search", "marketplace ranking",
    "recommendation system", "recommendation engine",
    "retrieval system", "retrieval engine", "matching system",
    "query understanding", "query rewriting",
    "re-ranking", "reranking", "re-ranker", "reranker",
    "hybrid search", "dense retrieval", "sparse retrieval",
    "semantic search", "vector search",
    "personalization system", "content discovery",
}

_RETRIEVAL_TECHNICAL = {
    "ltr", "recsys", "bm25", "tf-idf", "tfidf", "solr", "lucene",
    "re-ranking", "reranking", "re-ranker", "reranker",
}

_RETRIEVAL_GENERIC = {
    "retrieval", "search", "ranking", "recommendation", "recommender",
    "personalization", "matching", "relevance",
}

_RETRIEVAL_KEYWORDS = _RETRIEVAL_EXPLICIT | _RETRIEVAL_TECHNICAL | _RETRIEVAL_GENERIC


def retrieval_relevance_score(cand):
    """
    The single most important engineered feature.
    Measures evidence of production retrieval / search / ranking / recommendation work.
    """
    career = _career_text(cand)
    profile = cand.get("profile", {})
    profile_text = " ".join([
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]).lower()
    skills_text = " ".join(s.get("name", "") for s in cand.get("skills", [])).lower()
    evidence_text = career + " " + profile_text

    explicit_hits = _count_keyword_hits(evidence_text, _RETRIEVAL_EXPLICIT)
    technical_hits = _count_keyword_hits(evidence_text, _RETRIEVAL_TECHNICAL)
    generic_hits = _count_keyword_hits(career, _RETRIEVAL_GENERIC)
    skill_hits = _count_keyword_hits(skills_text, _RETRIEVAL_KEYWORDS)

    score = (
        explicit_hits * 0.30
        + technical_hits * 0.22
        + min(generic_hits, 3) * 0.08
        + min(skill_hits, 3) * 0.05
    )

    if explicit_hits == 0 and technical_hits == 0 and generic_hits == 0:
        score = min(score, 0.25)

    return min(1.0, score)


# ──────────────────────────────────────────────
# B. PRODUCT COMPANY SCORE
# ──────────────────────────────────────────────

_PRODUCT_COMPANY_KEYWORDS = {
    "startup", "start-up", "saas", "marketplace", "consumer tech",
    "consumer product", "b2b saas", "b2c", "product engineering",
    "flipkart", "swiggy", "razorpay", "zomato", "ola", "meesho",
    "uber", "airbnb", "doordash", "instacart", "stripe", "cred",
    "gojek", "grab", "agoda", "booking.com"
}

def product_company_score(cand):
    """
    Rewards experience in product startups, SaaS, marketplaces, etc.
    Applied as a positive signal rather than a hard requirement.
    """
    career = _career_text(cand)
    hits = _count_keyword_hits(career, _PRODUCT_COMPANY_KEYWORDS)
    
    # Also check company names explicitly
    for job in cand.get("career_history", []):
        company = job.get("company", "").lower()
        if any(kw in company for kw in _PRODUCT_COMPANY_KEYWORDS):
            hits += 2
            
    # Normalize: 2 hits = 0.5, 4 hits = 1.0
    score = min(1.0, hits / 4.0)
    return score


# ──────────────────────────────────────────────
# C. PRODUCT EXECUTION SCORE
# ──────────────────────────────────────────────

_EXECUTION_KEYWORDS = {
    "built", "shipped", "deployed", "launched", "productionized",
    "scaled", "implemented", "delivered", "architected", "designed and built",
    "production", "production system", "production environment",
    "end-to-end", "real-time", "real time",
    "microservice", "api", "pipeline",
    "serving", "inference", "latency",
    "users", "customers", "traffic",
    "millions", "thousands",
}


def product_execution_score(cand):
    """
    Rewards evidence of shipping real systems, not just studying them.
    The JD explicitly values 'shipper over researcher'.
    """
    career = _career_text(cand)
    summary = cand.get("profile", {}).get("summary", "").lower()
    combined = career + " " + summary

    hits = _count_keyword_hits(combined, _EXECUTION_KEYWORDS)

    # Normalize: 2 hits = 0.25, 5 hits = 0.625, 8+ hits = saturates
    score = min(1.0, hits / 8.0)
    return score


# ──────────────────────────────────────────────
# C. EVALUATION SYSTEMS SCORE
# ──────────────────────────────────────────────

_EVAL_STRONG = {
    "ndcg", "mrr", "map", "mean average precision",
    "a/b test", "a/b testing", "ab test", "ab testing",
    "offline evaluation", "online evaluation",
    "search metrics", "ranking metrics",
    "experimentation", "experiment framework",
    "relevance evaluation", "relevance judgment",
    "evaluation framework", "eval framework",
    "ranking evaluation", "search evaluation",
}

_EVAL_SUPPORTING = {
    "precision", "recall", "f1", "benchmark", "ground truth",
    "click-through rate", "ctr",
    "conversion rate",
}

_EVAL_KEYWORDS = _EVAL_STRONG | _EVAL_SUPPORTING


def evaluation_system_score(cand):
    """
    The JD references evaluation frameworks (NDCG, MRR, MAP, A/B testing).
    Requires stronger evidence from career descriptions, achievements, and summaries
    rather than isolated skills to reduce false positives.
    """
    career = _career_text(cand)
    summary = cand.get("profile", {}).get("summary", "").lower()
    combined = career + " " + summary
    
    # Also check accomplishments if present
    for job in cand.get("career_history", []):
        for acc in job.get("accomplishments", []):
            combined += " " + acc.lower()
            
    strong_hits = _count_keyword_hits(combined, _EVAL_STRONG)
    supporting_hits = _count_keyword_hits(combined, _EVAL_SUPPORTING)

    score = strong_hits * 0.32 + min(supporting_hits, 3) * 0.08
    if strong_hits == 0:
        score = min(score, 0.24)
    return min(1.0, score)


# ──────────────────────────────────────────────
# E. VECTOR SEARCH SCORE
# ──────────────────────────────────────────────

_VECTOR_PRODUCTS = {
    "faiss", "pinecone", "milvus", "qdrant", "weaviate",
    "elasticsearch", "opensearch", "elastic search",
}

_VECTOR_CONCEPTS = {
    "vector database", "vector db", "vector store",
    "vector search", "vector index",
    "hybrid search", "dense retrieval",
    "embedding index", "ann", "approximate nearest neighbor",
    "hnsw", "ivf",
}

_VECTOR_KEYWORDS = _VECTOR_PRODUCTS | _VECTOR_CONCEPTS


def vector_search_score(cand):
    """Rewards hands-on experience with vector databases and search infrastructure."""
    career = _career_text(cand)
    summary = cand.get("profile", {}).get("summary", "").lower()
    evidence_text = career + " " + summary
    skills_text = " ".join(s.get("name", "") for s in cand.get("skills", [])).lower()

    evidence_products = _count_keyword_hits(evidence_text, _VECTOR_PRODUCTS)
    evidence_concepts = _count_keyword_hits(evidence_text, _VECTOR_CONCEPTS)
    skill_products = _count_keyword_hits(skills_text, _VECTOR_PRODUCTS)
    skill_concepts = _count_keyword_hits(skills_text, _VECTOR_CONCEPTS)

    score = (
        evidence_products * 0.30
        + evidence_concepts * 0.22
        + min(skill_products, 3) * 0.12
        + min(skill_concepts, 2) * 0.06
    )

    if evidence_products == 0 and evidence_concepts == 0:
        score = min(score, 0.42)

    return min(1.0, score)


def feature_evidence(cand):
    """Return concrete text evidence used by recruiter-facing explanations."""
    career = _career_text(cand)
    profile = cand.get("profile", {})
    summary = profile.get("summary", "").lower()
    accomplishments = []
    for job in cand.get("career_history", []):
        accomplishments.extend(str(item).lower() for item in job.get("accomplishments", []))
    evidence_text = " ".join([career, summary, *accomplishments])

    return {
        "retrieval_relevance": _matched_keywords(
            evidence_text, _RETRIEVAL_EXPLICIT | _RETRIEVAL_TECHNICAL
        ),
        "evaluation_systems": _matched_keywords(evidence_text, _EVAL_STRONG),
        "vector_search": _matched_keywords(
            evidence_text, _VECTOR_PRODUCTS | _VECTOR_CONCEPTS
        ),
        "product_execution": _matched_keywords(evidence_text, _EXECUTION_KEYWORDS),
    }


# ──────────────────────────────────────────────
# F. NLP / LLM SCORE  (useful but NOT dominant)
# ──────────────────────────────────────────────

_NLP_KEYWORDS = {
    "nlp", "natural language processing", "natural language",
    "llm", "large language model",
    "embeddings", "embedding",
    "semantic search",
    "rag", "retrieval-augmented generation", "retrieval augmented generation",
    "fine-tuning", "fine tuning", "finetuning",
    "lora", "qlora", "peft",
    "transformer", "attention mechanism",
    "bert", "gpt", "t5",
    "hugging face", "huggingface",
    "sentence-transformers", "sentence transformers",
    "tokenization", "tokenizer",
    "text classification", "named entity",
    "sentiment analysis",
}


def nlp_llm_score(cand):
    """
    NLP/LLM expertise is valuable but the JD warns against buzzword-heavy profiles.
    This feature is intentionally capped to prevent domination.
    """
    text = _gather_text(cand)
    hits = _count_keyword_hits(text, _NLP_KEYWORDS)

    # Softer normalization — caps earlier to prevent domination
    score = min(1.0, hits / 8.0)
    return score


# ──────────────────────────────────────────────
# G. TITLE RELEVANCE
# ──────────────────────────────────────────────

_STRONG_TITLES = {
    "ml engineer", "machine learning engineer",
    "ai engineer", "applied ai engineer",
    "applied scientist", "applied ml scientist",
    "search engineer", "relevance engineer",
    "recommendation engineer", "ranking engineer",
    "nlp engineer", "data scientist",
    "ml scientist", "research engineer",
    "deep learning engineer",
}

_MODERATE_TITLES = {
    "backend engineer", "software engineer", "platform engineer",
    "data engineer", "full stack engineer", "fullstack engineer",
    "infrastructure engineer", "sre", "devops engineer",
    "senior engineer", "staff engineer", "principal engineer",
    "tech lead", "engineering manager",
}

_NEGATIVE_TITLES = {
    "marketing", "sales", "hr", "recruiter", "recruitment",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "chemical engineer", "structural engineer",
    "business analyst", "business development",
    "content writer", "graphic designer",
    "account manager", "project manager",
    "operations manager", "supply chain",
}


def title_relevance_score(cand):
    """
    Tiered scoring of current and recent titles.
    Prevents keyword-stuffing profiles with non-technical titles from ranking high.
    """
    profile = cand.get("profile", {})
    current_title = profile.get("current_title", "").lower()
    headline = profile.get("headline", "").lower()

    # Also check most recent career title
    career = cand.get("career_history", [])
    recent_title = career[0].get("title", "").lower() if career else ""

    combined = current_title + " " + headline + " " + recent_title

    # Check negative first
    for neg in _NEGATIVE_TITLES:
        if neg in combined:
            return 0.1

    # Check strong positive
    best = 0.4  # baseline for unrecognized tech titles
    for strong in _STRONG_TITLES:
        if strong in combined:
            best = max(best, 1.0)
    for mod in _MODERATE_TITLES:
        if mod in combined:
            best = max(best, 0.65)

    return best


# ──────────────────────────────────────────────
# H. EXPERIENCE ALIGNMENT
# ──────────────────────────────────────────────

def experience_alignment_score(cand):
    """
    JD target: 5-9 years. Good: 4-11. Moderate decay outside.
    Does NOT hard-penalize senior candidates — JD explicitly allows exceptions.
    """
    rel_years = cand.get("relevant_years", 0.0)
    total_years = cand.get("profile", {}).get("years_of_experience", 0)
    years = max(rel_years, total_years * 0.7)  # Use whichever gives a better signal

    if 5.0 <= years <= 9.0:
        return 1.0
    elif 4.0 <= years < 5.0:
        return 0.80
    elif 9.0 < years <= 11.0:
        return 0.80
    elif 11.0 < years <= 15.0:
        return 0.55
    elif 3.0 <= years < 4.0:
        return 0.45
    elif years > 15.0:
        return 0.40  # Soft — JD allows exceptions for strong profiles
    else:
        return 0.20  # < 3 years


# ──────────────────────────────────────────────
# I. CAREER STABILITY
# ──────────────────────────────────────────────

def career_stability_score(cand):
    """
    Computes average tenure and penalizes frequent job-hopping.
    JD Line 42: 'switching companies every 1.5 years, we're not a fit'.
    """
    career = cand.get("career_history", [])
    if not career:
        return 0.3

    durations = [job.get("duration_months", 0) for job in career]
    if not durations:
        return 0.3

    avg_tenure = sum(durations) / len(durations)
    short_stints = sum(1 for d in durations if d < 12)

    # Base score from average tenure
    if avg_tenure >= 36:
        score = 1.0
    elif avg_tenure >= 24:
        score = 0.85
    elif avg_tenure >= 18:
        score = 0.55
    elif avg_tenure >= 12:
        score = 0.35
    else:
        score = 0.15

    # Additional penalty for multiple short stints
    if short_stints >= 3:
        score *= 0.6
    elif short_stints >= 2:
        score *= 0.8

    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────
# J. ACTIVITY SCORE
# ──────────────────────────────────────────────

def activity_score(cand):
    """
    Uses behavioral signals to measure how 'alive' a candidate is on the platform.
    JD: 'a perfect-on-paper candidate who hasn't logged in for 6 months... not actually available.'
    """
    signals = cand.get("redrob_signals", {})

    # Recency (most important)
    last_active = signals.get("last_active_date", "")
    recency_score = 0.3  # default if unparseable
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_ago = (datetime.now() - last_dt).days
            if days_ago <= 14:
                recency_score = 1.0
            elif days_ago <= 30:
                recency_score = 0.9
            elif days_ago <= 60:
                recency_score = 0.75
            elif days_ago <= 90:
                recency_score = 0.55
            elif days_ago <= 180:
                recency_score = 0.30
            else:
                recency_score = 0.10
        except (ValueError, TypeError):
            pass

    # Engagement signals
    views = min(50, signals.get("profile_views_received_30d", 0)) / 50.0
    searches = min(100, signals.get("search_appearance_30d", 0)) / 100.0
    saved = min(20, signals.get("saved_by_recruiters_30d", 0)) / 20.0

    # Weighted combination — recency dominates
    score = recency_score * 0.50 + views * 0.20 + searches * 0.15 + saved * 0.15
    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────
# K. AVAILABILITY SCORE
# ──────────────────────────────────────────────

def availability_score(cand):
    """
    Rewards candidates who are actively available for hiring.
    JD: sub-30-day notice preferred, can buy out up to 30 days.
    """
    signals = cand.get("redrob_signals", {})

    # Open to work flag
    is_open = signals.get("open_to_work_flag", False)
    notice = signals.get("notice_period_days", 0)

    # Notice period scoring
    if notice <= 15:
        notice_score = 1.0  # Strong boost
    elif notice <= 30:
        notice_score = 0.8  # Positive
    elif notice <= 45:
        notice_score = 0.5  # Small penalty
    elif notice <= 60:
        notice_score = 0.2  # Moderate penalty
    else:
        notice_score = 0.05 # Meaningful penalty for 90+ days

    # Open-to-work boost
    open_score = 0.8 if is_open else 0.3

    # Response rate
    response_rate = signals.get("recruiter_response_rate", 0.5)

    score = notice_score * 0.45 + open_score * 0.30 + response_rate * 0.25
    return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────
# L. CONSISTENCY / HONEYPOT DETECTION
# ──────────────────────────────────────────────

def consistency_score(cand):
    """
    Trust score: detects honeypot and inflated profiles.
    Returns 1.0 for clean profiles, lower for suspicious ones.
    """
    penalties = 0.0

    skills = cand.get("skills", [])
    career = cand.get("career_history", [])
    profile = cand.get("profile", {})

    # 1. Expert skills with zero duration_months
    expert_zero_dur = 0
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", -1) == 0:
            expert_zero_dur += 1
    if expert_zero_dur >= 3:
        penalties += 0.35
    elif expert_zero_dur >= 1:
        penalties += 0.15

    # 2. Unrealistic experience claims (stated >> calculated)
    total_months = sum(job.get("duration_months", 0) for job in career)
    calculated_years = total_months / 12.0
    stated_years = profile.get("years_of_experience", 0)
    if stated_years > (calculated_years + 2.0):
        penalties += 0.30
    elif stated_years > (calculated_years + 0.5):
        penalties += 0.10

    # 3. Excessive expert skills with weak career
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 8 and calculated_years < 4:
        penalties += 0.30
    elif expert_count >= 6 and calculated_years < 3:
        penalties += 0.25

    # 4. Many skills with zero endorsements (potential stuffing)
    if len(skills) >= 10:
        zero_endorse = sum(1 for s in skills if s.get("endorsements", 0) == 0)
        if zero_endorse / len(skills) > 0.8:
            penalties += 0.10

    # 5. Very high skill count with generic names
    if len(skills) >= 20:
        penalties += 0.10

    return max(0.0, 1.0 - penalties)


# ──────────────────────────────────────────────
# M. RESEARCH PENALTY
# ──────────────────────────────────────────────

_RESEARCH_TITLES = {
    "research scientist", "research fellow", "research associate",
    "postdoctoral", "postdoc", "phd researcher", "phd student",
    "academic researcher", "research intern", "research assistant",
    "principal researcher", "senior researcher",
}

_PRODUCTION_EVIDENCE = {
    "deployed", "production", "shipped", "launched", "scaled",
    "real-time", "api", "microservice", "pipeline",
    "users", "customers", "revenue", "traffic",
    "productionized", "built and deployed",
}


def research_penalty(cand):
    """
    Soft penalty for pure research profiles without production evidence.
    JD: 'If you've spent your career in pure research environments without any production
    deployment — we will not move forward.'
    Does NOT penalize based on title alone — only when research-heavy AND lacks production work.
    """
    career = cand.get("career_history", [])
    if not career:
        return 0.0

    # Count research-titled roles
    research_roles = 0
    total_roles = len(career)
    for job in career:
        title = job.get("title", "").lower()
        if any(rt in title for rt in _RESEARCH_TITLES):
            research_roles += 1

    if research_roles == 0:
        return 0.0  # No research titles, no penalty

    research_ratio = research_roles / total_roles

    # Check for production evidence across all descriptions
    all_desc = _career_text(cand)
    production_hits = _count_keyword_hits(all_desc, _PRODUCTION_EVIDENCE)

    # Only penalize if research-heavy AND lacks production evidence
    if research_ratio >= 0.7 and production_hits <= 1:
        return 0.60  # Strong penalty — nearly pure research
    elif research_ratio >= 0.5 and production_hits <= 2:
        return 0.30  # Moderate penalty
    elif research_ratio >= 0.3 and production_hits == 0:
        return 0.15  # Mild penalty

    return 0.0  # Has enough production evidence


# ──────────────────────────────────────────────
# N. SPECIALIZATION MISMATCH PENALTY
# ──────────────────────────────────────────────

_CV_SPEECH_ROBOTICS = {
    "computer vision", "image recognition", "object detection",
    "image segmentation", "image classification",
    "speech recognition", "speech synthesis", "speech processing",
    "asr", "tts", "voice recognition",
    "robotics", "robot", "autonomous vehicle", "self-driving",
    "lidar", "slam", "motion planning",
}

_NLP_RETRIEVAL_EVIDENCE = {
    "nlp", "natural language", "retrieval", "search", "ranking",
    "recommendation", "information retrieval", "text",
    "llm", "language model", "embeddings",
    "recsys", "recommender",
}


def specialization_mismatch_penalty(cand):
    """
    Soft penalty for CV/Speech/Robotics specialists without NLP/IR exposure.
    JD Line 45: 'People whose primary expertise is computer vision, speech, or robotics
    without significant NLP/IR exposure... you'd be re-learning fundamentals.'
    """
    text = _gather_text(cand)

    cv_hits = _count_keyword_hits(text, _CV_SPEECH_ROBOTICS)
    
    # Include retrieval keywords in the NLP/IR evidence check to ensure
    # we don't penalize someone with both CV and solid retrieval experience
    combined_evidence = _NLP_RETRIEVAL_EVIDENCE.union(_RETRIEVAL_KEYWORDS)
    nlp_hits = _count_keyword_hits(text, combined_evidence)

    if cv_hits == 0:
        return 0.0

    # Only penalize if CV/Speech/Robotics is dominant AND NLP/IR is absent
    if cv_hits >= 4 and nlp_hits <= 1:
        return 0.50
    elif cv_hits >= 2 and nlp_hits == 0:
        return 0.35
    elif cv_hits >= 3 and nlp_hits <= 2:
        return 0.15

    return 0.0


# ──────────────────────────────────────────────
# O. CONSULTING PENALTY
# ──────────────────────────────────────────────

_CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "mphasis", "l&t infotech", "lti", "ltimindtree",
    "deloitte", "ey", "kpmg", "pwc",
}


def consulting_penalty(cand):
    """
    Graduated penalty for consulting-only careers.
    JD Line 44: 'People who have only worked at consulting firms in their entire career.'
    If candidate later moved into product companies, penalty is reduced or removed.
    """
    career = cand.get("career_history", [])
    if not career:
        return 0.0

    consulting_roles = 0
    product_roles = 0

    for job in career:
        company = job.get("company", "").lower()
        if any(firm in company for firm in _CONSULTING_FIRMS):
            consulting_roles += 1
        else:
            product_roles += 1

    if consulting_roles == 0:
        return 0.0

    total = len(career)
    consulting_ratio = consulting_roles / total

    # If they've moved into product companies, reduce penalty
    if product_roles >= 2:
        return 0.0  # Sufficient product experience — no penalty
    elif product_roles == 1:
        if consulting_ratio > 0.7:
            return 0.15  # Mostly consulting but has one product role
        return 0.0

    # Entire career is consulting
    if consulting_ratio >= 0.9:
        return 0.45
    elif consulting_ratio >= 0.7:
        return 0.25

    return 0.0


# ──────────────────────────────────────────────
# Master function: compute all features at once
# ──────────────────────────────────────────────

def compute_all_features(cand):
    """
    Computes all feature scores and penalties for a candidate in a single call.
    Attaches them directly to the candidate dict and returns the dict.
    """
    cand["feat_retrieval_relevance"] = retrieval_relevance_score(cand)
    cand["feat_product_company"] = product_company_score(cand)
    cand["feat_product_execution"] = product_execution_score(cand)
    cand["feat_evaluation_systems"] = evaluation_system_score(cand)
    cand["feat_vector_search"] = vector_search_score(cand)
    cand["feat_nlp_llm"] = nlp_llm_score(cand)
    cand["feat_title_relevance"] = title_relevance_score(cand)
    cand["feat_experience_alignment"] = experience_alignment_score(cand)
    cand["feat_career_stability"] = career_stability_score(cand)
    cand["feat_activity"] = activity_score(cand)
    cand["feat_availability"] = availability_score(cand)
    cand["feat_consistency"] = consistency_score(cand)

    # Penalties
    cand["pen_research"] = research_penalty(cand)
    cand["pen_specialization"] = specialization_mismatch_penalty(cand)
    cand["pen_consulting"] = consulting_penalty(cand)

    return cand
