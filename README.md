# AURA: CachedRAG Cost-Optimized Document Intelligence Engine

**AURA** is an advanced, production-grade Retrieval-Augmented Generation (RAG) system engineered to drastically reduce LLM API computation costs and latency. Instead of invoking a remote LLM for every query, AURA integrates a local **Semantic Cache Layer** (simulating Redis Vector Similarity Search) and local in-memory vector indexing, backed by a **Dual-LLM Cross-Evaluation Pattern**.

---

## 🚀 Key Architectural Features

1. **Semantic Cache Layer (Redis VSS Simulation)**:
   - Evaluates natural language queries using local vector similarity checks (Cosine Similarity).
   - If a similar question was previously asked (above a configurable similarity threshold, e.g., `0.85`), AURA returns the answer **instantly (< 10ms)**, bypassing the RAG pipeline and saving 100% of LLM generation costs.

2. **Knowledge Retrieval (Qdrant In-Memory)**:
   - Indexes uploaded documents (PDFs, TXT) into a local in-memory Qdrant database, eliminating the need for complex external database setups for quick deployments.

3. **High-Speed Inference (Groq + Llama 3.1 8B)**:
   - When a cache miss occurs, the RAG pipeline is triggered, and Groq's high-speed free developer tier executes inference using `llama-3.1-8b-instant`.

4. **Cross-Evaluation Judge (Google Gemini 2.5 Flash)**:
   - Automatically audits primary LLM outputs for correctness. Calculates **Faithfulness** (hallucination detection) and **Relevance** scores, logging results in a real-time audit trail and updating the cache.

5. **Real-time Analytics Dashboard (React + Vite)**:
   - Visualizes cost savings, token usage, real-time cache hit ratios, and latency comparisons (Cache < 10ms vs LLM > 1000ms).
   - Shows Gemini Alignment Judge scorecards and live audit logs.

---

## 🛠️ Tech Stack

- **Frontend**: React.js, Vite, Lucide-React, Pure CSS (custom glassmorphic theme).
- **Backend Routing**: FastAPI, Uvicorn, Python Multipart.
- **Vector Embeddings**: Local `sentence-transformers` (`all-MiniLM-L6-v2`) with a deterministic hashing trick fallback.
- **Vector Database**: `qdrant-client` running in local in-memory mode.
- **Evaluation Judge**: Google Gemini API (`gemini-2.5-flash`).
- **Primary Inference**: Groq Cloud LLM (`llama-3.1-8b-instant`).

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Node.js v18+ & npm

### 1. Backend Configuration
Navigate to the `backend/` directory:
```bash
cd backend
```

Create a virtual environment and activate it:
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

Create a `.env` file from the template and configure your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
SEMANTIC_CACHE_THRESHOLD=0.85
EVALUATION_RATE=1.0
```
> **Note**: If API keys are omitted, the server will launch in **Simulation Mode** using high-fidelity mock generators/evaluators so the UI can be run and demonstrated fully offline.

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
The backend API will run on `http://localhost:8000`.

### 2. Frontend Configuration
Navigate to the `frontend/` directory:
```bash
cd ../frontend
```

Install packages:
```bash
npm install
```

Start the Vite development server:
```bash
npm run dev
```
Open your browser and navigate to `http://localhost:5173`.

---

## 📊 Evaluation & Verification Flow

1. **Ingest Docs**: Drag and drop a `.txt` or `.pdf` file. Look at the left panel to confirm the file is indexed.
2. **First Query (Cache Miss)**: Ask a question (e.g. *"What is the main topic of document?"*).
   - Watch the RAG pipeline retrieve chunks, call Groq for synthesis, and trigger Gemini for audit checks.
   - Response latency: ~1,000–1,500ms.
3. **Second Query (Cache Hit)**: Ask a highly similar question (e.g. *"Can you tell me the main topic of the document?"*).
   - The semantic cache intercepts the request.
   - Response latency: **< 10ms** (Instant).
4. **Dashboard View**: Monitor live cache hit ratios, token metrics, total cost accumulated, and estimated USD cost saved.
