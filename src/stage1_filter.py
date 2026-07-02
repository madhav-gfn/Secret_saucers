import json
from datetime import datetime, timedelta
from src.config import NOTICE_PERIOD_TRAP_DAYS, MIN_EXPERIENCE_YEARS

# JD Line 44: Explicit consulting firm blocklist
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "mphasis", "l&t infotech", "lti", "ltimindtree"
}

# JD Line 48/60: Indian cities where the role is based
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
    "bangalore", "bengaluru", "gurgaon", "gurugram", "chennai"
}

def stream_and_filter_candidates(file_path):
    """
    Reads the candidates file (handles both .json sample array and .jsonl full file).
    Applies Stage 1 fast pruning with DUAL strategy:
      - Hard-drops clearly disqualified candidates to keep Stage 2 embedding fast
      - Borderline cases pass through but get soft penalties in Stage 3
    Yields passing candidates with location_score and relevant_years pre-computed.
    """
    file_path_str = str(file_path)
    
    if file_path_str.endswith(".jsonl"):
        # Full dataset (100k)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                cand = json.loads(line)
                if passes_baseline_filters(cand):
                    yield cand
    else:
        # Sample dataset
        with open(file_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
            for cand in candidates:
                if passes_baseline_filters(cand):
                    yield cand

def passes_baseline_filters(cand):
    """
    Hard filters to keep candidate pool manageable for CPU embedding in Stage 2.
    
    Strategy: aggressively prune clearly unfit candidates here, then apply
    graduated soft penalties in features.py / Stage 3 for borderline cases
    that survive this stage.
    """
    career_history = cand.get("career_history", [])
    profile = cand.get("profile", {})
    signals = cand.get("redrob_signals", {})

    # ── Compute relevant experience (needed downstream) ──
    tech_keywords = {
        "engineer", "developer", "software", "data", "ml", "ai",
        "backend", "frontend", "fullstack", "scientist", "analytics",
        "programmer", "architect", "devops", "sre", "platform",
        "research", "nlp", "machine learning", "deep learning",
    }
    
    relevant_months = 0
    total_months = 0
    for job in career_history:
        dur = job.get("duration_months", 0)
        total_months += dur
        title = job.get("title", "").lower()
        if any(kw in title for kw in tech_keywords):
            relevant_months += dur
            
    relevant_years = relevant_months / 12.0
    calculated_years = total_months / 12.0
    
    cand["relevant_years"] = relevant_years
    cand["calculated_years"] = calculated_years

    # ── HARD FILTER 1: Minimum relevant experience ──
    if relevant_years < MIN_EXPERIENCE_YEARS:
        return False
        
    # ── HARD FILTER 2: Honeypot — stated >> calculated experience ──
    stated_years = profile.get("years_of_experience", 0)
    if stated_years > (calculated_years + 1.5):
        # Tighter than before (1.5 year gap) — catches more honeypots
        # Mild mismatches (0.5-1.5 gap) still get soft penalty via consistency_score
        return False
    
    # ── HARD FILTER 3: Current title is clearly non-technical ──
    current_title = profile.get("current_title", "").lower()
    non_tech_signals = {"marketing manager", "sales manager", "hr manager",
                        "recruiter", "business development", "civil engineer",
                        "mechanical engineer", "content writer", "accountant",
                        "project manager", "business analyst"}
    if any(sig in current_title for sig in non_tech_signals):
        return False

    # Also check: if current title has NO tech keyword at all, drop
    if current_title and not any(kw in current_title for kw in tech_keywords):
        return False

    # ── HARD FILTER 4: Notice period extreme ──
    if signals.get("notice_period_days", 0) > NOTICE_PERIOD_TRAP_DAYS:
        return False

    # ── HARD FILTER 5: Activity Recency (JD Line 67) ──
    # "hasn't logged in for 6 months... not actually available"
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            cutoff = datetime.now() - timedelta(days=180)
            if last_dt < cutoff:
                return False
        except (ValueError, TypeError):
            pass

    # ── HARD FILTER 6: Consulting-Only Career (JD Line 44) ──
    # Only drops if ENTIRE career is consulting. Mixed careers pass through
    # and get a soft penalty in features.py
    if career_history:
        all_consulting = True
        for job in career_history:
            company = job.get("company", "").lower()
            if not any(firm in company for firm in CONSULTING_FIRMS):
                all_consulting = False
                break
        if all_consulting:
            return False

    # ── HARD FILTER 7: Job-Hopper (extreme cases only) ──
    # Only drops if 4+ jobs averaging < 15 months. Moderate hopping (3 jobs, 18 mo)
    # passes through and gets soft penalty.
    if len(career_history) >= 4:
        avg_tenure = total_months / len(career_history)
        if avg_tenure < 15:
            return False

    # ── HARD FILTER 8: Behavioral dead-end ──
    is_open = signals.get("open_to_work_flag", False)
    recruiter_views = signals.get("profile_views_received_30d", 0)
    if not is_open and recruiter_views == 0:
        return False
        
    # ── HARD FILTER 9: Offer acceptance trap ──
    if "offer_acceptance_rate" in signals and signals["offer_acceptance_rate"] < 0.20:
        return False

    # ── Tag location data for Stage 3 scoring ──
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)
    
    if any(city in location for city in PREFERRED_LOCATIONS):
        cand["location_score"] = 1.0
    elif "india" in country:
        cand["location_score"] = 0.7 if relocate else 0.5
    else:
        cand["location_score"] = 0.3 if relocate else 0.1
        
    return True
