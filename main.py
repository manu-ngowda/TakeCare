"""
main.py
───────
Hospital Triage RAG — Cloud Run Backend
FastAPI server that exposes the RAG pipeline as HTTP endpoints.

Deploy with:
    gcloud run deploy triage-agent \
        --source . \
        --region asia-south1 \
        --allow-unauthenticated \
        --set-env-vars GEMINI_API_KEY=<your-key>
"""

import os
import json
import time
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_pipeline import TriageRAGPipeline, intake_chat

# ─────────────────────────────────────────────
# 0. LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. APP LIFESPAN — build vector index ONCE at startup
#    (not on every request — embedding 25 docs is expensive)
# ─────────────────────────────────────────────

pipeline: Optional[TriageRAGPipeline] = None
_pipeline_lock = threading.Lock()


def _build_pipeline_background() -> None:
    """Build embeddings in a background thread so uvicorn binds to PORT immediately."""
    global pipeline
    kb_path = os.getenv("KB_PATH", "knowledge_base.json")
    try:
        log.info("Background: building RAG vector index (this may take 1–2 min) …")
        built = TriageRAGPipeline(knowledge_base_path=kb_path)
        with _pipeline_lock:
            pipeline = built
        log.info("RAG pipeline ready.")
    except Exception as exc:
        log.error("Pipeline build failed: %s", exc, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the HTTP server immediately; build the vector index in the background.
    Cloud Run requires the process to listen on PORT before the startup timeout.
    """
    threading.Thread(target=_build_pipeline_background, daemon=True).start()
    log.info("Server listening — pipeline initialising in background.")
    yield
    log.info("Shutting down.")


# ─────────────────────────────────────────────
# 2. FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Hospital Triage RAG Agent",
    description="Gemini-powered triage assistant using Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan
)

# Allow requests from any frontend (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# 3. REQUEST / RESPONSE SCHEMAS  (Pydantic)
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Full conversation history including the latest user message",
    )


class CollectedPatient(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    chief_complaint: Optional[str] = None
    symptoms: Optional[str] = None
    onset: Optional[str] = None
    allergies: Optional[str] = None
    medications: Optional[str] = None
    medical_history: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    intake_complete: bool = False
    collected: CollectedPatient = Field(default_factory=CollectedPatient)
    symptoms_summary: str = ""


class TriageRequest(BaseModel):
    symptoms: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Patient's symptom description in plain English",
        example="Patient has crushing chest pain, sweating, and left arm numbness."
    )
    patient_id: Optional[str] = Field(
        None,
        description="Optional patient ID for logging purposes"
    )


class RetrievedSource(BaseModel):
    title: str
    score: float
    id: Optional[str] = None
    department: Optional[str] = None
    priority: Optional[int] = None
    semantic_score: Optional[float] = None
    tfidf_score: Optional[float] = None
    term_overlap: Optional[float] = None
    sanitized: Optional[bool] = None


class TriageResponse(BaseModel):
    department: str
    priority: int
    priority_label: str
    rationale: str
    confidence: str
    floor: str = "1"
    room: str = "101"
    retrieved_sources: list[RetrievedSource]
    processing_time_ms: int
    matched_protocol_id: Optional[str] = None
    retrieval_top_score: Optional[float] = None


# ─────────────────────────────────────────────
# 4. ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — used by Cloud Run to verify the container is alive."""
    return {"status": "ok", "service": "Hospital Triage RAG Agent"}


@app.get("/health")
def health():
    """Detailed health check including pipeline status."""
    return {
        "status": "healthy" if pipeline else "initialising",
        "pipeline_ready": pipeline is not None,
        "knowledge_base": os.getenv("KB_PATH", "knowledge_base.json")
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Conversational patient intake — Gemini asks follow-up questions like a
    healthcare assistant until enough information is collected for triage.
    """
    log.info(f"Intake chat | turns={len(request.messages)}")
    try:
        result = intake_chat([m.model_dump() for m in request.messages])
    except Exception as e:
        log.error(f"Intake chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Intake chat error: {str(e)}")

    collected = result.get("collected") or {}
    return ChatResponse(
        reply=result.get("reply", "Could you tell me a bit more about how you're feeling?"),
        intake_complete=bool(result.get("intake_complete")),
        collected=CollectedPatient(**{k: collected.get(k) for k in CollectedPatient.model_fields}),
        symptoms_summary=result.get("symptoms_summary", ""),
    )


@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    """
    Main endpoint — receives patient symptoms, runs the full RAG pipeline,
    and returns a structured triage decision.

    Pipeline steps (all happen inside pipeline.triage()):
      1. Embed symptoms          → vector representation
      2. Semantic search         → top-3 relevant clinical documents
      3. Inject context          → augmented prompt for Gemini
      4. Generate               → structured JSON triage decision
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet. Retry in a few seconds.")

    log.info(f"Triage request | patient_id={request.patient_id} | symptoms='{request.symptoms[:60]}…'")

    start = time.time()
    try:
        result = pipeline.triage(request.symptoms)
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Triage pipeline error: {str(e)}")

    elapsed_ms = int((time.time() - start) * 1000)
    log.info(f"Triage complete in {elapsed_ms}ms — {result['department']} | Priority {result['priority']}")

    return TriageResponse(
        department=result["department"],
        priority=result["priority"],
        priority_label=result["priority_label"],
        rationale=result["rationale"],
        confidence=result["confidence"],
        floor=str(result.get("floor", "1")),
        room=str(result.get("room", "101")),
        retrieved_sources=[RetrievedSource(**s) for s in result["retrieved_sources"]],
        processing_time_ms=elapsed_ms,
        matched_protocol_id=result.get("matched_protocol_id"),
        retrieval_top_score=result.get("retrieval_top_score"),
    )


@app.get("/rag/retrieve")
async def rag_retrieve(symptoms: str):
    """
    Debug endpoint — inspect hybrid retrieval scores without calling Gemini.
    Tune RAG_MIN_SCORE, RAG_TOP_K, RAG_KEYWORD_BOOST via environment variables.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet.")
    if len(symptoms.strip()) < 5:
        raise HTTPException(status_code=400, detail="symptoms query must be at least 5 characters.")
    return pipeline.retrieve_only(symptoms)


@app.get("/knowledge-base")
def get_knowledge_base():
    """
    Returns the list of documents in the knowledge base (without embeddings).
    Useful for debugging and the presentation demo.
    """
    kb_path = os.getenv("KB_PATH", "knowledge_base.json")
    try:
        with open(kb_path) as f:
            docs = json.load(f)
        # Strip the embedding vectors — they're huge and not human-readable
        clean = [
            {k: v for k, v in doc.items() if k != "embedding"}
            for doc in docs
        ]
        return {"count": len(clean), "documents": clean}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Knowledge base file not found.")


# ─────────────────────────────────────────────
# 5. GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."}
    )


# ─────────────────────────────────────────────
# 6. LOCAL DEV ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
