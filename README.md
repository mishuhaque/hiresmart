# 🤖 HireSmart – AI Candidate Ranker

🚀 **HireSmart** is an **AI-native system** for ranking interview candidates.  
Instead of relying on rigid scoring rules, outdated ML heuristics, or inconsistent human judgment, HireSmart uses **RAG (Retrieval-Augmented Generation) + LangChain** to evaluate and rank candidates based on **past hiring outcomes**.  

It ingests candidate profiles (25+ features: communication, technical skills, adaptability, problem solving, teamwork, etc.), retrieves similar past candidates, and generates **explainable hire/not hire decisions** — along with **executive-style justifications**.  

---

## ✨ Why This Matters
- ⏱️ **Save HR time** – No more manual shortlisting across hundreds of candidates.  
- 📊 **Consistent & objective** – Removes human bias and subjective scoring.  
- 💡 **Explainable AI** – Provides top-K similar past candidates as reasoning.  
- 🔮 **Scalable** – Handles 100s or 1000s of candidates across hiring pipelines.  

---

## 🆚 Why HireSmart is Superior
### ❌ Traditional Rule-Based Ranking
- Manually crafted scoring weights.  
- Inflexible to new candidate traits.  
- No reasoning or justification.  

### ❌ Basic ML Ranking
- Requires engineered features + labeled data.  
- Produces probabilities but lacks transparency.  
- Struggles to adapt to qualitative human traits.  

### ❌ Human Screening
- Inconsistent across interviewers.  
- Time-consuming and expensive.  
- Prone to bias and fatigue.  

### ✅ HireSmart (RAG + LangChain)
- **Retrieves** top-K most similar past candidates (vector search).  
- **Aggregates** decisions (HIRE/NOT HIRE) for consistency.  
- **Generates** natural language explanations using LLMs (e.g. Flan-T5, Mistral).  
- **Adapts** seamlessly when new hiring data is added.  

---

## 🛠️ Tech Stack
- **LangChain** → RAG pipeline for candidate retrieval.  
- **FAISS** → Vector similarity search over candidate embeddings.  
- **Sentence Transformers** → Embeddings (`all-MiniLM-L6-v2`).  
- **Transformers / Flan-T5** → AI-generated justifications.  
- **Python, Pandas** → Data handling + pipelines.  

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data (past hires)
python src/generate_text_training_data.py

# 3. Build FAISS index from training dataset
python src/rag_candidate_ranker.py

# 4. Query one candidate (demo)
python src/query_candidate.py

# 5. Batch evaluate a CSV of candidates
python src/evaluate_candidates.py

# 6. Extract Top-5 recommended hires
python src/select_top5.py


src/
├── generate_text_training_data.py   # Build synthetic training dataset
├── rag_candidate_ranker.py          # Build FAISS index
├── query_candidate.py               # Query one candidate
├── evaluate_candidates.py           # Evaluate a CSV of candidates
├── select_top5.py                   # Select Top-5 hires
data/
├── training_text_candidates.csv     # Synthetic training dataset
├── candidate_index/                 # FAISS vector index
├── interview_candidates.csv         # New candidates to evaluate
└── evaluation_results.csv           # AI decisions + explanations



🎯 New Candidate Evaluation
Profile: Candidate Profile: good technical skills, weak communication, reliable problem solving, strong adaptability

Top Similar Past Candidates:
- Candidate Profile: strong technical skills, reliable problem solving, weak willing to learn | Label: HIRE
- Candidate Profile: strong task prioritization, excellent technical skills | Label: HIRE
- Candidate Profile: reliable technical skills, excellent creativity, outstanding adaptability | Label: HIRE

✅ Decision: HIRE
📝 Explanation: This candidate is recommended due to strong technical skills and adaptability, consistent with past successful hires.
