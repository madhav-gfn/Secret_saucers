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
    Applies Stage 1 fast pruning (Honeypot & Baseline Filter).
    Yields passing candidates.
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
    Applies strict heuristic baseline filters inspired by VitaSort & HRClaw,
    plus explicit JD-mandated disqualifiers.
    """
    # 1. Notice Period Trap Filter
    signals = cand.get("redrob_signals", {})
    if signals.get("notice_period_days", 0) > NOTICE_PERIOD_TRAP_DAYS:
        return False
        
    # 2. Relevant Experience Filter (Ignore non-tech roles like HR/Sales)
    career_history = cand.get("career_history", [])
    tech_keywords = {"engineer", "developer", "software", "data", "ml", "ai", "backend", "frontend", "fullstack", "scientist", "analytics", "programmer"}
    
    relevant_months = 0
    total_months = 0
    for job in career_history:
        dur = job.get("duration_months", 0)
        total_months += dur
        title = job.get("title", "").lower()
        # If any tech keyword is in the title, count as relevant
        if any(kw in title for kw in tech_keywords):
            relevant_months += dur
            
    relevant_years = relevant_months / 12.0
    calculated_years = total_months / 12.0
    
    if relevant_years < MIN_EXPERIENCE_YEARS:
        return False
        
    cand["relevant_years"] = relevant_years
        
    # 2b. Experience Mismatch Trap (Honeypot detection)
    profile = cand.get("profile", {})
    stated_years = profile.get("years_of_experience", 0)
    if stated_years > (calculated_years + 0.25):
        return False
        
    # 2c. Current Title Check (JD Line 66)
    # "A candidate whose title is 'Marketing Manager' is not a fit"
    current_title = profile.get("current_title", "").lower()
    if current_title and not any(kw in current_title for kw in tech_keywords):
        return False
    
    # ---- NEW: JD-MANDATED DISQUALIFIERS ----
    
    # 3. Consulting Company Trap (JD Line 44)
    # Filter candidates who have ONLY worked at consulting firms
    if career_history:
        all_consulting = True
        for job in career_history:
            company = job.get("company", "").lower()
            if not any(firm in company for firm in CONSULTING_FIRMS):
                all_consulting = False
                break
        if all_consulting:
            return False
    
    # 4. Activity Recency (JD Line 67)
    # "hasn't logged in for 6 months... not actually available"
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d")
            cutoff = datetime.now() - timedelta(days=180)
            if last_dt < cutoff:
                return False
        except (ValueError, TypeError):
            pass  # If date is unparseable, don't filter
    
    # 5. Title-Chaser / Job-Hopper Detection (JD Line 42)
    # "switching companies every 1.5 years, we're not a fit"
    if len(career_history) >= 3:
        avg_tenure = total_months / len(career_history)
        if avg_tenure < 18:  # Less than 1.5 years average
            return False
    
    # 6. Behavioral & Conversion Dead-End Filter
    is_open = signals.get("open_to_work_flag", False)
    recruiter_views = signals.get("profile_views_received_30d", 0)
    if not is_open and recruiter_views == 0:
        return False
        
    # 6b. Conversion Trap: If they reject almost every offer, don't waste time
    if "offer_acceptance_rate" in signals and signals["offer_acceptance_rate"] < 0.20:
        return False
    
    # 7. CV/Speech/Robotics without NLP/IR Trap (JD Line 45)
    # Check if they are heavily CV/Speech/Robotics focused
    has_cv_speech = False
    has_nlp_ir = False
    cv_keywords = {"vision", "cv", "speech", "robotics", "image"}
    nlp_keywords = {"nlp", "ir", "search", "retrieval", "ranking", "recommendation", "language", "llm", "text"}
    
    for job in career_history:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        combined = title + " " + desc
        
        if any(kw in combined for kw in cv_keywords):
            has_cv_speech = True
        if any(kw in combined for kw in nlp_keywords):
            has_nlp_ir = True
            
    # Also check explicitly stated skills
    for s in cand.get("skills", []):
        name = s.get("name", "").lower()
        if any(kw in name for kw in cv_keywords):
            has_cv_speech = True
        if any(kw in name for kw in nlp_keywords):
            has_nlp_ir = True
            
    if has_cv_speech and not has_nlp_ir:
        return False
        
    # 8. Tag location data for Stage 3 scoring (JD Line 48/60)
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)
    
    if any(city in location for city in PREFERRED_LOCATIONS):
        cand["location_score"] = 1.0  # In a preferred city
    elif "india" in country:
        cand["location_score"] = 0.7 if relocate else 0.5  # In India but not preferred city
    else:
        cand["location_score"] = 0.3 if relocate else 0.1  # Outside India
        
    return True
