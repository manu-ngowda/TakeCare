# Hospital Triage RAG Agent

An intelligent, GenAI-powered healthcare triage system. This project uses **Retrieval-Augmented Generation (RAG)** to provide conversational patient intake and automated triage routing based on a synthetic clinical knowledge base.

It consists of a FastAPI backend connecting to the Gemini API, and a beautiful front-end interface (`takecare_wired.html`) for interacting with the system.

---

## 🌟 Key Features

1. **Conversational Patient Intake (`/chat`)**
   - Uses Gemini to conduct a natural, empathetic interview (collecting name, symptoms, timeline, and allergies).
2. **Intelligent Triage (`/triage`)**
   - Implements a hybrid RAG pipeline (Semantic Search + TF-IDF + Keyword Overlap) to match patient symptoms to the most relevant clinical protocols.
   - Recommends the appropriate hospital department, priority level, and provides a rationale grounded *only* in the retrieved documents.
3. **Security Guardrails**
   - Includes document and query sanitization to protect against prompt injections.
4. **Interactive UI**
   - A single-page application (`takecare_wired.html`) offering a seamless interface for the intake chat, viewing triage results, and generating clinical handoff notes.

---

## 📁 Project Structure

```text
triage-agent/
├── main.py               ← FastAPI server (Cloud Run entrypoint)
├── rag_pipeline.py       ← Core RAG logic (embed → search → inject → respond)
├── rag_config.py         ← Configuration parameters for the RAG pipeline
├── rag_rerank.py         ← Hybrid reranker implementation
├── rag_sanitize.py       ← Prompt-injection security scanning
├── knowledge_base.json   ← Synthetic clinical documents used for grounding
├── takecare_wired.html   ← Frontend web interface
├── Dockerfile            ← Container definition for Cloud Run
├── requirements.txt      ← Python dependencies
└── README.md             ← This file
```

---

## 🚀 Local Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API Key
The application requires a valid Gemini API key to run.
```bash
# PowerShell
$env:GEMINI_API_KEY="your-key-here"

# CMD
set GEMINI_API_KEY=your-key-here

# Linux/Mac
export GEMINI_API_KEY="your-key-here"
```

### 3. Run the Server
Start the FastAPI server locally:
```bash
python main.py
```
*The server will start at `http://localhost:8080`. The vector index for the knowledge base builds in the background during startup.*

### 4. Open the Frontend
Once the server is running, simply open `takecare_wired.html` in any web browser to interact with the system via the UI. 
Make sure the frontend is configured to point to your local server if testing locally.

---

## 🧪 Testing the API

You can test the core `/triage` endpoint directly using cURL:

```bash
curl -X POST http://localhost:8080/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "Patient has crushing chest pain and sweating, radiating to the left arm."}'
```

**Example Response:**
```json
{
  "department": "Cardiology Emergency",
  "priority": 1,
  "priority_label": "Critical",
  "rationale": "Symptoms strongly match myocardial infarction protocols.",
  "confidence": "high",
  "floor": "2",
  "room": "204",
  "retrieved_sources": [
    { "title": "Myocardial Infarction Protocol", "score": 0.942, "id": "KB001" }
  ],
  "processing_time_ms": 1420
}
```

---

## ☁️ Deploy to Google Cloud Run

Deploy the backend directly to Google Cloud Run from your terminal:

```bash
gcloud run deploy triage-agent \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --timeout 300 \
  --cpu-boost \
  --set-env-vars GEMINI_API_KEY=your-key-here
```

*Note: The server responds immediately, but the in-memory RAG index takes ~1–2 minutes to build in the background. You can poll `GET /health` to verify when `"pipeline_ready": true`.*

---

## ⚙️ Fine-Tuning the RAG Pipeline

You can tweak the retrieval logic using environment variables (defined in `rag_config.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAG_TOP_K` | 5 | Max number of protocols to inject into the Gemini prompt |
| `RAG_MIN_SCORE` | 0.42 | Minimum semantic similarity score for a document to be considered |
| `RAG_KEYWORD_BOOST` | 0.08 | Weight applied for symptom keyword overlap |
| `RAG_STRONG_MATCH` | 0.72 | Threshold at which the output aligns directly with the top KB document |
| `RAG_GEN_TEMP` | 0.1 | Temperature used for the Gemini JSON generation |

**Hybrid Reranking Weights:**
- `RAG_RERANK_SEMANTIC`: 0.50
- `RAG_RERANK_TFIDF`: 0.35
- `RAG_RERANK_OVERLAP`: 0.15

---

## 🏗️ Architecture Overview

```text
Patient Symptoms (Text Input)
        │
        ▼
[1] EMBEDDING (Gemini Embeddings API)
        │  Generates query vector
        ▼
[2] HYBRID SEARCH & RERANKING
        │  Cosine similarity + TF-IDF + Keyword Overlap
        │  Returns top relevant clinical docs from Knowledge Base
        ▼
[3] CONTEXT INJECTION (Prompt Engineering)
        │  System rules + Retrieved Documents + Sanitized Symptoms
        ▼
[4] GENERATION (Gemini Pro/Flash)
        │  Outputs grounded, structured JSON
        ▼
Triage Decision (Department, Priority, Rationale, Location)
```
