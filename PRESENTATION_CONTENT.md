# 🚀 Intelligent Candidate Discovery & Ranking — Presentation Content

---

## Slide 1: Solution Overview

### What is the proposed solution?

- A **4-stage, CPU-only** candidate ranking pipeline that processes **100,000 candidates in ~40 seconds**
- Stages:
  1. **Viability Pruning** — fast hard filters to drop clearly unfit candidates
  2. **Semantic + Feature Extraction** — dense embeddings + 12 domain-engineered features
  3. **12-Feature Scorecard Ranking** — weighted aggregation with capped penalties
  4. **Evidence-Backed Explanation Generation** — recruiter-style, fact-grounded reasoning

### What differentiates it from traditional matching?

- **Beyond Keywords:** Combines semantic embeddings with 12 domain-engineered features (retrieval expertise, product execution, evaluation frameworks, career stability, etc.)
- **Trap-Aware:** Explicitly detects and evades **7+ hidden disqualifiers** — honeypot profiles, keyword stuffers, negative-mention traps (LangChain, CV/Speech/Robotics)
- **Dual-Filter Design:** Hard pruning for efficiency + graduated soft penalties for nuance — no good candidate is blindly dropped
- **Zero-Hallucination Explanations:** Rule-based, evidence-backed reasoning — no LLM generation in the output

---

## Slide 2: JD Understanding & Candidate Evaluation

### Key Requirements Extracted from JD

- **Core Domain:** Information Retrieval, Search, Ranking, Recommendation Systems
- **Mindset:** "Shippers over researchers" — values production deployment evidence
- **Experience Band:** 5–9 years (sweet spot), with soft decay outside
- **Evaluation Skills:** NDCG, MRR, MAP, A/B testing — rare & highly valued
- **Red Flags:** Consulting-only careers, CV/Speech/Robotics without NLP, LangChain-tutorial GitHub, keyword-stuffed profiles

### How candidate fit is evaluated beyond keywords

- **Semantic Embeddings** (summary + top 2 career descriptions) compared via cosine similarity against JD
- **Weighted Jaccard** skill matching against tokenized JD (with stopword/negative-mention blocklist)
- **12 Engineered Features** assess depth across: retrieval relevance, product execution, evaluation systems, vector search infra, NLP/LLM expertise, title relevance, experience alignment, career stability, platform activity, availability, product-company background, and profile consistency
- **Honeypot Detection:** Flags stated experience exceeding calculated career timeline

---

## Slide 3: Ranking Methodology

### How candidates are retrieved, scored, and ranked

| Stage | Action | Output |
|-------|--------|--------|
| **Stage 1** | Stream 100k JSONL → apply 9 hard filters | ~4% survive (~4,000 candidates) |
| **Stage 2** | Encode via SentenceTransformer → FAISS search → extract 12 features | Scored candidate pool |
| **Stage 3** | Build scorecard → apply penalties → sort descending | Top 100 ranked candidates |

### Models & Algorithms

- **`all-MiniLM-L6-v2`** (ONNX backend, CPU) — lightweight 384-dim sentence embeddings
- **`FAISS IndexFlatIP`** — exact cosine similarity via inner product of normalized vectors
- **Weighted Jaccard** with proficiency-aware scoring:
  - Advanced = 1.5×, Intermediate = 1.2×, High endorsements = 1.2× multiplier

### How multiple signals are combined into a final ranking

- **Blended Semantic** = 65% embedding + 35% hard-skill Jaccard (within the semantic weight slot)
- **Weighted Sum** across 13 features:

  | Feature | Weight |
  |---------|--------|
  | Semantic (blended) | 16% |
  | Retrieval Relevance | 15% |
  | Product Execution | 12% |
  | Evaluation Systems | 10% |
  | Career Stability | 8% |
  | Experience Alignment | 7% |
  | Title Relevance | 6% |
  | Activity | 6% |
  | Availability | 5% |
  | Product Company | 5% |
  | Vector Search | 4% |
  | NLP/LLM | 3% |
  | Consistency | 3% |

- **3 Capped Penalties** subtracted:

  | Penalty | Cap |
  |---------|-----|
  | Research (pure academic, no production) | 8% |
  | Specialization mismatch (CV/Speech/Robotics) | 6% |
  | Consulting-only career | 6% |

- **Location Bonus** added (2% max, based on preferred Indian cities / relocation willingness)
- **Final Score** = Base Weighted Sum − Penalty Total + Location Bonus (clamped 0–1)

---

## Slide 4: Explainability & Data Validation

### How ranking decisions are explained

- **Deterministic, template-driven** text generation — no LLM involved
- Each candidate triggers the **top 2 strongest features** via activation thresholds (e.g., retrieval ≥ 0.60, evaluation ≥ 0.50)
- Explanations cite **specific facts**: current title, company, years of experience, matched skills, career trajectory
- **Tone adapts to rank:**
  - Top 10 → "Strong match"
  - Top 30 → "Good match"
  - Top 60 → "Moderate match"
  - 60+ → "Borderline match"
- Both **strengths and tradeoffs** are always included (e.g., _"60-day notice period exceeds JD preference"_)

### How hallucinations are prevented

- Every claim is backed by extracted data arrays — `feature_evidence()` returns matched keywords from actual career text
- Display engine uses a **`TECHNICAL_CATEGORIES` whitelist** — prevents showing noise skills like "Marketing" or "Content Writing"
- Evidence terms are **deduplicated** across explanation sections to avoid repetition
- No generative model is used for output text — all reasoning is rule-based

### How inconsistent / suspicious profiles are handled

| Check | Action |
|-------|--------|
| Stated experience > calculated timeline by >1.5 years | **Hard drop** (honeypot) |
| Offer acceptance rate < 20% | **Hard drop** |
| Expert skills with 0 duration months | **Soft penalty** via `consistency_score` |
| Excessive expert count with short career | **Soft penalty** via `consistency_score` |
| 80%+ zero-endorsement skill arrays | **Soft penalty** (skill stuffing) |
| Career-wide consulting without product transition | **Soft penalty** via `consulting_penalty` |
| Not logged in for 6+ months | **Hard drop** |

---

## Slide 5: End-to-End Workflow

```
JD Text File (input) + 100k Candidate JSONL (input)
           │
   ┌───────▼────────────────────────────────┐
   │  STAGE 1: Viability Pruning            │
   │  9 hard filters → ~96% dropped         │
   │  (honeypots, non-tech, inactive,       │
   │   extreme notice, job-hoppers)         │
   └───────┬────────────────────────────────┘
           │  ~4,000 survivors
   ┌───────▼────────────────────────────────┐
   │  STAGE 2: Semantic + Feature           │
   │  MiniLM embedding + FAISS search       │
   │  + Weighted Jaccard skill matching     │
   │  + 12 domain feature extraction        │
   └───────┬────────────────────────────────┘
           │
   ┌───────▼────────────────────────────────┐
   │  STAGE 3: Scorecard Ranking            │
   │  Weighted sum − penalties + bonus      │
   │  → Sort descending → Select Top 100   │
   └───────┬────────────────────────────────┘
           │
   ┌───────▼────────────────────────────────┐
   │  STAGE 4: Explanation Generation       │
   │  Evidence-backed reasoning text        │
   │  + Debug CSVs for internal validation  │
   └───────┬────────────────────────────────┘
           │
           ▼
   submission.csv        → Top 100 ranked (candidate_id, rank, score, reasoning)
   submission_debug.csv  → All feature scores for every candidate
   top20_debug.csv       → Deep diagnostic for top 20
```

---

## Slide 6: System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Orchestrator)                   │
│         Wires all 4 stages together, controls pipeline flow     │
└──────┬──────────┬──────────────┬──────────────┬─────────────────┘
       │          │              │              │
       ▼          ▼              ▼              ▼
  ┌─────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────────┐
  │ Stage 1 │ │ Stage 2  │ │  Stage 3     │ │     Stage 4      │
  │ Filter  │ │ Ensemble │ │  Scorecard   │ │   Explanation    │
  └────┬────┘ └────┬─────┘ └──────┬───────┘ └────────┬─────────┘
       │           │              │                   │
       │           │         ┌────▼──────┐            │
       │           │         │ features  │            │
       │           │         │   .py     │            │
       │           │         │ (818 LOC) │            │
       │           │         │ 12 scores │            │
       │           │         │ 3 penalty │            │
       │           │         └───────────┘            │
       │           │                                  │
       ▼           ▼                                  ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                      config.py                              │
  │  Feature weights (13) · Penalty caps (3) · Thresholds      │
  │  Data paths · Hard-skill blend ratio · Filter parameters   │
  └─────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | File | Responsibility |
|--------|------|----------------|
| **Config** | `config.py` | Feature weights, penalty caps, data paths, thresholds |
| **Data Ingestion + Filter** | `stage1_filter.py` | Streaming JSONL reader, 9 hard filters, location tagging |
| **Vector Engine** | `stage2_ensemble.py` | SentenceTransformer (ONNX), FAISS Index, Weighted Jaccard |
| **Feature Engine** | `features.py` | 12 domain scorers, 3 penalty functions, evidence extractor |
| **Ranking** | `stage3_scorecard.py` | Blended scoring, weighted aggregation, sorting |
| **Explainer** | `stage4_explanation.py` | Threshold activation, recruiter-style text, CSV generation |
| **Orchestrator** | `main.py` | Pipeline controller — end-to-end execution |
| **Utilities** | `utils.py` | RAKE-based JD keyword extraction (NLTK) |

---

## Slide 7: Results & Performance

### Ranking Quality

- Successfully surfaces candidates with **production retrieval/search/ranking** experience to the top
- Internal diagnostic CSVs expose all **12 feature scores + 3 penalties** for every ranked candidate
- Feature distribution diagnostics (mean, median, min, max) printed for Top 100
- Explanation activation counts verified — ensures diverse, non-boilerplate reasoning
- **100% compliant** with challenge's `validate_submission.py`

### Meeting Runtime & Compute Constraints

| Metric | Value |
|--------|-------|
| **Dataset Size** | 100,000 JSONL records |
| **Total Runtime** | ~40 seconds on standard CPU |
| **Time Limit** | 300 seconds (5 minutes) |
| **Hardware** | Standard CPU, 16 GB RAM |
| **External API Calls** | Zero — fully local |

### Key Optimizations

- **Stage 1 prunes ~96%** before any embedding runs — the single biggest speed win
- **ONNX backend** for SentenceTransformer eliminates PyTorch overhead at inference
- **Batch encoding** (batch_size=256) maximizes CPU throughput
- **CPU-only PyTorch** build (`torch==2.3.1+cpu`) keeps dependency footprint minimal
- **Generator-based streaming** — 100k JSONL records are never fully loaded into memory

---

## Slide 8: Technologies Used

| Technology | Purpose | Why Selected |
|---|---|---|
| **Python 3** | Core language | Rich ML/NLP ecosystem |
| **SentenceTransformers** (`all-MiniLM-L6-v2`) | Dense text embeddings (384-dim) | Lightweight, accurate, CPU-friendly |
| **ONNX Runtime + Optimum** | Model inference backend | 2–3× faster than raw PyTorch on CPU |
| **FAISS** (`faiss-cpu`) | Vector similarity search (`IndexFlatIP`) | Sub-second cosine search over thousands of vectors |
| **PyTorch** (CPU-only build) | ML framework dependency | Required by SentenceTransformers; CPU build avoids CUDA overhead |
| **Transformers** (HuggingFace) | Tokenizer and model loading | Industry-standard model hub |
| **Pandas** | DataFrame manipulation & CSV export | Clean tabular output for submission + debug files |
| **NLTK + RAKE-NLTK** | JD keyword extraction | Automatic technical phrase extraction from raw JD text |
| **scikit-learn** | Utility ML functions | Available for additional feature engineering |
| **NumPy** | Numerical array operations | Efficient math for embeddings and scoring |
| **JSON streaming + Generators** | Memory-efficient data loading | Handles 100k records without loading all into memory |

---

> **Note:** All technologies are free/open-source. No paid services, external APIs, or GPU hardware are required.
