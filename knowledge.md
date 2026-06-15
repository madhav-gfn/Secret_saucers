Relevant Open-Source Projects
Several GitHub projects tackle resume screening and ranking that can be adapted. For example, Toshan Kanwar’s Resume-Ranking-System combines multiple matching methods (BERT and MiniLM embeddings, TF-IDF, Jaccard skill overlap, spaCy NER) into a weighted ensemble to score resumes. Abhishek Singh’s ResumeRanker AI uses Sentence-Transformers (all-MiniLM-L6-v2) to compute cosine similarity between a job description and each resume. VitaSort (by la-b-ib) uses TF-IDF vectorization and cosine similarity to match resumes, augmented with interactive visualizations (radar charts, word clouds, skill heatmaps) for analysis. The resume-rag-ranker project (alexcfv) demonstrates a two-stage pipeline: ingest PDF resumes into a ChromaDB vector store using embeddings, then use an LLM to build structured profiles and re-rank the top results. JENITH47’s skillScout pipeline highlights BERT embeddings combined with machine‐learning ensembles (XGBoost/LightGBM/CatBoost) and bias-aware scoring, with a Streamlit dashboard for recruiters. Finally, HRClaw (qinjobs) is a local-first screening toolkit that builds a JD “scorecard”, bulk-imports resumes (PDF/DOCX), extracts structured profiles, and assigns evidence-backed scores (recommend/reject). These projects cover key functions (semantic matching, batch scoring, explainability) but would need extensions to incorporate our specific signals (notice period, geo-flexibility, activity scores) and to meet the strict 5-minute, CPU-only constraint.

Project Requirements and Goals
The system must rank 100,000 candidates and output the top 100 with a score and a 1–2 sentence reasoning for each. Deep understanding of the Senior AI Engineer JD is needed – the model should infer required years of AI/ML experience, product vs. service background, production ML systems, vector DB knowledge, etc., beyond mere keyword matching. Crucially, it must integrate behavioral signals: candidate availability (e.g. “open to work” flag, notice-period days), recent activity (logins, recruiter views, GitHub commits), and responsiveness (response rate, interview/offer rates). The scoring should boost ready-to-move candidates and penalize those with long notice periods or low engagement. The output is a ranked shortlist of 100, with transparent justifications for each choice. For example, modern AI rankers emphasize explainable scores and “why” analyses for each candidate. We must also avoid obvious traps (honeypot profiles with impossible timelines, title-chasers, pure academics) by using sanity checks on career history. Finally, the solution must run in ≤5 minutes on a 16 GB CPU machine with no external API calls.

Multi-Stage Ranking Pipeline
A multi-stage funnel is ideal for balancing speed and accuracy. Early stages use fast heuristics to prune unsuitable resumes, and later stages apply heavier models to a smaller set. For example:

Stage 1 – Heuristic Filtering (100k→~10k): Quickly drop clearly mismatched profiles. Examples: eliminate anyone with <4 total years of experience, or with only consulting/service companies in history, or zero mention of AI/ML; drop profiles inactive for >1 year or with 0% recruiter response. We can also detect “honeypots” by flagging impossible data (e.g. 8-year tenure at a 3-year-old company) and exclude them. These checks run in seconds on CPU.

Stage 2 – Semantic Retrieval (10k→~1k): Embed each remaining resume (e.g. profile summary + key skills) and the job description into vectors using a fast sentence embedding model (such as all-MiniLM-L6-v2). Index the 10k candidate vectors with an in-memory vector store (e.g. FAISS-CPU with inner-product or cosine similarity). Query by the JD embedding to retrieve the top ~1,000 candidates by semantic relevance. This hybrid lexical+vector search mimics Elasticsearch’s two-stage approach, where a broad first-pass (TF-IDF or k-NN) feeds candidates to a reranker.

Stage 3 – Feature Ranking (1k→100): For these ~1k candidates, extract rich features and score them with a machine-learning ranker. Features include: Profile features (total experience, AI-specific years, number of AI projects, education tier), skill overlap (Jaccard/Tf-idf vs. JD must-haves), company signals (past product-company tenures), vector search scores (embedding distance), and behavioral signals (notice_days, relocation flag, response_rate, GitHub_activity_score, etc.). A learning-to-rank model (e.g. XGBoost or LightGBM) can learn from labeled examples or heuristics. Note that Elastic’s docs recommend LTR when you have training data. Alternatively, a weighted scoring function can combine semantic score, skill match, and boosted signals. (For instance, Toshan’s system shows a weighted sum of BERT, SBERT, TF-IDF, Jaccard and an ML score – we would similarly weight our signals.) Honeypots and mismatches should be hard-penalized in this stage.

Stage 4 – Explanation Generation (Top 100): For each of the final 100, generate a concise justification referencing the profile facts and JD fit (or gap). This could be templated (“Has X years in ML, built Y, but notice period is Z”) or use a small local LLM (e.g. a quantized Llama/Mistral) to draft the sentence. The two-stage RAG example even uses an LLM to “explain” each top candidate. Since we have CPU only, we might use a distilled model or careful prompt engineering, but the key is to ensure factual grounding (no hallucinations) and mention both strengths and any concerns.

Throughout the pipeline we balance contextual matching (semantic vectors, LTR) with signal integration. This mirrors best practices: cheap initial filters plus more expensive ML-based reranking for final results. We must tune for speed (e.g. use faiss-cpu with 384-dim embeddings as in popular demos) so the whole job (100k→100) finishes in under 5 minutes on 16 GB.

AI Coder Implementation Plan
An “AI-assisted coder” can break this into concrete tasks and milestones:

Data Ingestion & Cleaning: Load the provided candidates.jsonl. Parse each JSON into structured records (profile, career history, skills, signals). Normalize dates and compute derived stats (total experience, AI-related years, etc.). Flag obvious data errors (e.g. negative durations). (Milestone: Clean dataset and summary stats.)

Heuristic Filter Module: Code filters to remove mismatches. For example, filter by years_of_experience < 4; exclude if current_company is in known large IT-service lists; drop if recruiter_response_rate == 0 and last_active_date is >6 months old. Also implement honeypot checks (e.g. sum of durations vs. total experience). (Milestone: Filtered candidate set, debug output of remaining count.)

Embedding & Vector Indexing: Use the sentence-transformers library (free) to load all-MiniLM-L6-v2. Generate embeddings for each candidate’s text blob (e.g. concatenated summary+skills) and for the JD. Install and initialize faiss-cpu to index the candidate vectors. Perform a top-N nearest-neighbors search of the JD vector to get ~1k candidates. (Milestone: Verified vector search returns sensible similar candidates.)

Feature Extraction: For the ~1k survivors, compute detailed features: binary flags (has_AI_ml_experience, worked_at_product_company), numeric (years_exp, skill_proficiency_scores), and use given signals (notice days, github score, activity metrics). Also include the FAISS cosine distance or BM25 score as a feature. Collate these into a feature matrix. (Milestone: Candidate feature table prepared.)

Ranking Model / Scoring: Decide on ranking method: either train a simple LTR model (if any labeled data or simulate with heuristics), or craft a custom scoring function (weighted sum). For example, one could train an XGBoost ranker where relevant candidates (based on some validation) are higher. Ensure strong penalties for long notice, missing requirements, or only “window dressing” skills. (Milestone: Ranking model scores computed for top candidates.)

Result Compilation: Select the top 100 by final score. Generate human-readable reasoning for each. This could use string templates (e.g., “X years at Y company with ML role, high recruiter engagement”) or a CPU-compatible LLM. For example, one might prompt a 2–4B Llama with “[Profile facts] The candidate is a good fit because… (mention skills and any caveats)”. Ensure no hallucinations by feeding factual bullet points. (Milestone: Completed candidate_id, rank, score, reasoning for 100 lines.)

Output Validation: Run validate_submission.py on the CSV to check format. Inspect a sample of reasoning outputs for fidelity. (Milestone: Validation passes.)

Each task can be developed incrementally, with unit tests. We can harness AI tools (e.g. Copilot) to help write code snippets for parsing, filtering, and model training, but all libraries and resources are free and must run locally.

Required Resources (Free/Open Tools)
All components can use free/open libraries and models:

Embeddings & Similarity: HuggingFace’s sentence-transformers (e.g. all-MiniLM-L6-v2 or similar) for text embeddings. Install via pip install sentence-transformers.
Vector Index: Facebook’s FAISS (faiss-cpu) for fast nearest-neighbor search.
Machine Learning: Scikit-learn, XGBoost or LightGBM (all free) for any LTR or scoring models.
NLP: spaCy or NLTK for any lightweight text processing (skill extraction, etc.).
Data Handling: Pandas, NumPy for data manipulation.
Local LLM (optional): If explanation text is generated with an LLM, use an open checkpoint (e.g. Llama-2-7B) that can run on CPU (or small distilled model like flan-t5-base). Else use templating logic.
Compute: A standard CPU instance with ~16 GB RAM (as allowed).
No external APIs: All models must be loaded locally (no OpenAI/Gemini calls).
No paid services or non-free licenses are needed. The candidate dataset is already provided by the challenge. The only “data” required beyond that are general-purpose text models, which are free.

Additional Considerations
Evaluation: Although no ground truth is given, we should test internally by verifying known good candidates (e.g. did experts from similar backgrounds rank higher). Use ranking metrics (NDCG, MRR) on any small labeled subset if available.
Bias & Fairness: Avoid using sensitive PII; focus on skills and signals. Optionally anonymize names/titles before modeling.
Performance Tuning: Profile memory/time. Optimize FAISS index parameters (e.g. IndexFlatIP) and batch embedding calls. Ensure the entire pipeline (data load → top-100 output) completes within the time limit.
Documentation: Comment code and prepare clear reasoning output. The challenge demands transparent “reasoning” text, so draft careful templates or prompts that cite actual profile details (experience years, companies, key skills, notice days).
By leveraging and adapting these open-source ideas and following a staged design, we can meet the challenge’s goals: deep semantic matching plus realistic hiring signals, all within the compute limits. This plan ensures we build on proven methods while adding the necessary domain-specific logic.

Sources: Example systems and techniques from GitHub (Resume-Ranking-System, ResumeRanker AI, VitaSort, resume-rag-ranker, skillScout, HRClaw) and best-practice references on ranked search pipelines and explainable resume scoring. These guided the above requirements and architecture.