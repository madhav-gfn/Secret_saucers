import faiss
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Common English stopwords that could appear as fake "skills"
# ALSO includes Negative Mentions explicitly rejected by the JD (Line 43/45)
STOPWORDS = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
             "was", "one", "our", "out", "has", "had", "how", "its", "may", "new",
             "now", "old", "see", "way", "who", "did", "get", "let", "say", "she",
             "too", "use", "set", "map", "run", "age", "art", "log", "ion", "arm",
             "langchain", "vision", "speech", "robotics"}

class EnsembleMatcher:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.jd_vector = None
        self.jd_text = ""
        self.jd_words = set()  # Word-boundary tokenized set for exact matching
        
    def load_jd(self, jd_path, dynamic_required=None, dynamic_rejected=None):
        with open(jd_path, "r", encoding="utf-8") as f:
            self.jd_text = f.read()
            
        # Tokenize JD into individual lowercase words (alphanumeric only)
        raw_tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', self.jd_text.lower())
        self.jd_words = set()
        
        # If the dynamic LLM succeeded, we ONLY use the words it extracted as positive signals
        if dynamic_required:
            for skill in dynamic_required:
                for word in re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', skill):
                    if len(word) >= 3:
                        self.jd_words.add(word)
        else:
            # Fallback: tokenizing the whole document
            for token in raw_tokens:
                self.jd_words.add(token)
                # Split hyphenated compounds: "data-infra" -> "data", "infra"
                if '-' in token:
                    for part in token.split('-'):
                        if len(part) >= 3:
                            self.jd_words.add(part)
        
        # Apply Stopwords / Blocklist
        self.jd_words -= STOPWORDS
        
        # If the LLM successfully extracted negative skills, remove them too!
        if dynamic_rejected:
            for skill in dynamic_rejected:
                for word in re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', skill):
                    if word in self.jd_words:
                        self.jd_words.remove(word)
        
        self.jd_vector = self.model.encode([self.jd_text], normalize_embeddings=True)
        
    def compute_weighted_skills(self, skills_list):
        """
        Compute weighted Jaccard similarity dynamically:
        If a candidate's skill is explicitly mentioned in the JD, it counts as a match!
        """
        if not skills_list:
            return 0.0, []
            
        matched_names = []
        score = 0.0
        
        for s in skills_list:
            name = s.get("name", "")
            name_lower = name.lower().strip()
            # Skip very short or common words (stopwords like "the", "not", "set")
            if len(name_lower) < 3:
                continue
            # For multi-word skills like "Fine-tuning LLMs", check if ANY word appears in JD
            # For single-word skills, do exact word-boundary match against tokenized JD
            skill_tokens = set(re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', name_lower))
            matched_tokens = skill_tokens.intersection(self.jd_words)
            # Require at least one meaningful token to match (skip 1-2 char tokens and stopwords)
            meaningful_matches = [t for t in matched_tokens if len(t) >= 3 and t not in STOPWORDS]
            if meaningful_matches:
                matched_names.append(name)
                
                prof = s.get("proficiency", "beginner")
                endorses = s.get("endorsements", 0)
                
                # Base points based on proficiency
                pts = 1.0
                if prof == "advanced":
                    pts = 1.5
                elif prof == "intermediate":
                    pts = 1.2
                    
                # Multiplier for high endorsements
                if endorses > 20:
                    pts *= 1.2
                    
                score += pts
                
        # Normalize against a theoretical perfect score (e.g., matching 10 advanced skills = 10 * 1.8 = 18 points)
        max_possible = 18.0
        final_score = (score / max_possible) if max_possible > 0 else 0.0
        return final_score, matched_names
        
    def score_candidates(self, candidates):
        """
        Compute Semantic and Jaccard scores. TF-IDF is implicitly represented 
        by Jaccard for the exact keyword matches in our lightweight ensemble.
        """
        if not candidates:
            return []
            
        texts = []
        for c in candidates:
            profile = c.get("profile", {})
            skill_names = [s.get("name", "") for s in c.get("skills", [])]
            
            # Deep Context: Extract top 2 career history descriptions
            career_history = c.get("career_history", [])[:2]
            career_text = " ".join([job.get("description", "") for job in career_history])
            
            texts.append(f"{profile.get('summary', '')} {career_text} {' '.join(skill_names)}")
            
        cand_vectors = self.model.encode(texts, normalize_embeddings=True)
        
        # Inner product of normalized vectors = Cosine Similarity
        index = faiss.IndexFlatIP(cand_vectors.shape[1])
        index.add(cand_vectors)
        
        # Search JD against candidates
        k = len(candidates)
        similarities, indices = index.search(self.jd_vector, k)
        
        # Map semantic scores
        semantic_scores = {}
        for sim, idx in zip(similarities[0], indices[0]):
            semantic_scores[idx] = float(sim)
            
        scored_candidates = []
        for i, cand in enumerate(candidates):
            semantic = semantic_scores.get(i, 0.0)
            skills_data = cand.get("skills", [])
            weighted_score, matched_names = self.compute_weighted_skills(skills_data)
            
            # Bound values 0 to 1
            semantic = max(0.0, min(1.0, semantic))
            weighted_score = max(0.0, min(1.0, weighted_score))
            
            cand["semantic_score"] = semantic
            cand["hard_skill_score"] = weighted_score
            cand["matched_skills"] = matched_names
            scored_candidates.append(cand)
            
        return scored_candidates
