"""
stage1b_tfidf.py — Fast TF-IDF pre-filter between Stage 1 and Stage 2.

Narrows the candidate pool from ~7,500 to a target top-K (default 3,000)
before the expensive SentenceTransformer encoding in Stage 2.

Uses scikit-learn's TfidfVectorizer for cosine similarity against the JD.
Includes a safety net: candidates with strongly relevant titles are always
retained regardless of TF-IDF score, so we don't lose high-value candidates
who use different vocabulary.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Titles that indicate direct relevance to the JD — these candidates
# are force-included even if TF-IDF score is low (different vocabulary ≠ bad fit)
_FORCE_INCLUDE_TITLE_KEYWORDS = {
    "search engineer", "ranking engineer", "relevance engineer",
    "recommendation systems engineer", "recommendation engineer",
    "applied scientist", "applied ml", "applied ai",
    "ml engineer", "machine learning engineer", "ai engineer",
    "nlp engineer", "data scientist", "research engineer",
    "deep learning engineer", "staff machine learning",
    "senior machine learning", "lead ai", "lead ml",
}


def _build_candidate_text(cand):
    """Build a single text blob for TF-IDF from a candidate's profile."""
    profile = cand.get("profile", {})
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]
    # Career descriptions (all jobs, not just top 2 like Stage 2)
    for job in cand.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    # Skill names
    for s in cand.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(parts)


def tfidf_prefilter(candidates, jd_path, top_k=3000):
    """
    Fast TF-IDF pre-filter to narrow candidate pool before embedding.

    Args:
        candidates: list of candidate dicts (Stage 1 survivors)
        jd_path: path to job_description.txt
        top_k: number of candidates to retain (default 3000)

    Returns:
        list of candidate dicts, ordered by TF-IDF similarity (descending)
    """
    if len(candidates) <= top_k:
        print(f"  TF-IDF: Only {len(candidates)} candidates — no filtering needed.")
        return candidates

    # Load JD
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    # Build text for all candidates
    candidate_texts = [_build_candidate_text(c) for c in candidates]

    # Fit TF-IDF on all texts (JD + candidates)
    all_texts = [jd_text] + candidate_texts
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),   # Unigrams + bigrams for phrases like "vector search"
        stop_words="english",
        sublinear_tf=True,    # Apply log normalization to term frequency
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Cosine similarity: JD (index 0) vs all candidates (index 1+)
    jd_vector = tfidf_matrix[0:1]
    cand_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(jd_vector, cand_vectors).flatten()

    # Identify force-include candidates (strong titles that must survive)
    force_include_indices = set()
    for i, cand in enumerate(candidates):
        title = cand.get("profile", {}).get("current_title", "").lower()
        if any(kw in title for kw in _FORCE_INCLUDE_TITLE_KEYWORDS):
            force_include_indices.add(i)

    # Rank by TF-IDF similarity and take top_k
    ranked_indices = similarities.argsort()[::-1]  # Descending

    selected_indices = set()
    result = []

    # First: add force-included candidates
    for idx in force_include_indices:
        selected_indices.add(idx)

    # Then: fill remaining slots from TF-IDF ranking
    for idx in ranked_indices:
        if len(selected_indices) >= top_k:
            break
        selected_indices.add(idx)

    # Build result in TF-IDF rank order (force-includes first, then by similarity)
    # Store TF-IDF score on each candidate for potential downstream use
    for idx in ranked_indices:
        if idx in selected_indices:
            candidates[idx]["tfidf_score"] = float(similarities[idx])
            result.append(candidates[idx])

    # Add any force-includes that weren't already in the result
    for idx in force_include_indices:
        if idx not in {ranked_indices[i] for i in range(len(ranked_indices)) if ranked_indices[i] in selected_indices}:
            candidates[idx]["tfidf_score"] = float(similarities[idx])
            result.append(candidates[idx])

    force_count = len(force_include_indices - set(ranked_indices[:top_k]))
    print(f"  TF-IDF: {len(candidates)} → {len(result)} candidates "
          f"(top_k={top_k}, force-included {len(force_include_indices)} by title)")

    return result
