import nltk
from rake_nltk import Rake

# Ensure nltk data is downloaded (failsafe for first run)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def extract_jd_keywords(jd_text, top_n=20):
    """
    Extracts the highest scoring technical noun phrases from a JD text.
    Returns a set of lowercased strings.
    """
    # Initialize RAKE (limit phrases to 1 or 2 words to act like specific skills)
    r = Rake(min_length=1, max_length=2)
    r.extract_keywords_from_text(jd_text)
    
    # Get top phrases
    phrases = r.get_ranked_phrases()[:top_n]
    
    # Clean and split them so "python/pytorch" becomes "python", "pytorch"
    final_skills = set()
    for phrase in phrases:
        words = phrase.replace("/", " ").split()
        for w in words:
            final_skills.add(w.lower())
            
    return final_skills
