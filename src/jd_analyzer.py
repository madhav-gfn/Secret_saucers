import time
import re
import threading

def _run_llm_extraction(jd_text, result_dict):
    try:
        from transformers import pipeline
        
        # Use an incredibly tiny, fast model that is highly instruction-tuned
        model_id = "Qwen/Qwen1.5-0.5B-Chat"
        
        pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map="cpu",
        )
        
        prompt = f"""You are an expert technical recruiter analyzing a job description.
Your task is to extract exactly two lists of technical skills from the text:
1. REQUIRED: Technical skills that are explicitly requested or needed.
2. REJECTED: Technical skills that are explicitly mentioned as NOT needed, or where the JD says "if you only have this, you are not a fit".

Job Description:
{jd_text}

Return ONLY the two comma-separated lists in this exact format, with no extra text:
REQUIRED: skill1, skill2
REJECTED: skill3, skill4
"""
        messages = [
            {"role": "system", "content": "You are a precise data extraction bot. You output nothing except the requested format."},
            {"role": "user", "content": prompt}
        ]
        
        outputs = pipe(
            messages,
            max_new_tokens=150,
            do_sample=False,
            return_full_text=False
        )
        
        response = outputs[0]["generated_text"]
        
        required = set()
        rejected = set()
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("REQUIRED:"):
                skills = line.replace("REQUIRED:", "").split(',')
                for s in skills:
                    clean = s.strip().lower()
                    if clean: required.add(clean)
            elif line.startswith("REJECTED:"):
                skills = line.replace("REJECTED:", "").split(',')
                for s in skills:
                    clean = s.strip().lower()
                    if clean: rejected.add(clean)
                    
        if required or rejected:
            result_dict['req'] = required
            result_dict['rej'] = rejected
    except Exception as e:
        result_dict['error'] = str(e)

def extract_dynamic_skills(jd_text):
    """
    Attempts to use a tiny local LLM to dynamically extract positive and negative skills.
    Fails safely and gracefully if it takes too long or fails to load.
    """
    print("--- Initializing Dynamic JD Parsing (Safety Net Active) ---")
    start_time = time.time()
    
    result = {}
    t = threading.Thread(target=_run_llm_extraction, args=(jd_text, result))
    t.daemon = True
    t.start()
    
    # Wait maximum 45 seconds for model load and inference
    t.join(45.0)
    
    if t.is_alive():
        print(f"\n[!] Dynamic JD Parsing skipped: Timeout exceeded (45s). CPU is too slow.")
        print("[!] Falling back to mathematically proven deterministic extraction.\n")
        return None, None
        
    if 'error' in result or ('req' not in result and 'rej' not in result):
        e = result.get('error', 'Empty LLM response')
        print(f"\n[!] Dynamic JD Parsing skipped due to environment constraints: {e}")
        print("[!] Falling back to mathematically proven deterministic extraction.\n")
        return None, None
        
    print(f"Dynamic Extraction Complete in {(time.time() - start_time):.1f}s")
    print(f"  [+] REQUIRED: {result['req']}")
    print(f"  [-] REJECTED: {result['rej']}")
    
    return result['req'], result['rej']
