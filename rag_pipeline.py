"""
rag_pipeline.py
───────────────
Hospital Triage RAG Pipeline
Core flow: symptoms → embed → semantic search → inject context → Gemini responds

Concepts used:
  • Lab 5  – Embeddings & vector similarity (cosine similarity)
  • Lab 6  – Retrieval-Augmented Generation (RAG) architecture
  • Lab 7  – Prompt engineering & Gemini API integration
"""

import os
import re
import json
import math
from typing import List, Dict, Tuple, Optional

import google.generativeai as genai

from rag_config import (
    TOP_K,
    RETRIEVE_CANDIDATES,
    MIN_SIMILARITY,
    KEYWORD_BOOST,
    STRONG_MATCH_THRESHOLD,
    GENERATION_TEMPERATURE,
    RERANK_WEIGHT_SEMANTIC,
    RERANK_WEIGHT_TFIDF,
    RERANK_WEIGHT_OVERLAP,
    PRIORITY_LABELS,
    SYMPTOM_ALIASES,
    get_department_location,
)
from rag_sanitize import sanitize_document, sanitize_text
from rag_rerank import HybridReranker

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────

def get_gemini_api_key() -> str:
    """Read API key from any supported environment variable."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def ensure_genai_configured() -> None:
    """Configure the Gemini client or raise a clear setup error."""
    key = get_gemini_api_key()
    if not key:
        raise RuntimeError(
            "Gemini API key not found.\n"
            "  PowerShell:  $env:GEMINI_API_KEY = 'your-key-here'\n"
            "  CMD:         set GEMINI_API_KEY=your-key-here\n"
            "  Linux/Mac:   export GEMINI_API_KEY=your-key-here\n"
            "Also accepts GOOGLE_API_KEY."
        )
    genai.configure(api_key=key)


ensure_genai_configured()

# Tried in order until one works (override with GEMINI_EMBED_MODEL)
EMBED_MODEL_FALLBACKS = [
    "models/gemini-embedding-2",
    "models/embedding-001",
]
_active_embed_model: Optional[str] = None

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def resolve_embed_model() -> str:
    """Pick the first embedding model that works with this API key."""
    global _active_embed_model
    if _active_embed_model:
        return _active_embed_model

    explicit = os.getenv("GEMINI_EMBED_MODEL", "").strip()
    if explicit:
        _probe_embed_model(explicit)
        _active_embed_model = explicit
        print(f"[RAG] Embedding model (env): {explicit}")
        return explicit

    ensure_genai_configured()
    last_error: Optional[Exception] = None
    for model in EMBED_MODEL_FALLBACKS:
        try:
            _probe_embed_model(model)
            _active_embed_model = model
            print(f"[RAG] Embedding model: {model}")
            return model
        except Exception as e:
            last_error = e
            print(f"[RAG] Skipping {model}: {e}")

    raise RuntimeError(
        "No embedding model available. Set GEMINI_EMBED_MODEL to a valid model, e.g. "
        "models/embedding-001 or models/gemini-embedding-2."
    ) from last_error


def _probe_embed_model(model: str) -> None:
    """Test that a model accepts embed_content."""
    kwargs: Dict = {"model": model, "content": "triage health check"}
    if "embedding-001" in model:
        kwargs["task_type"] = "retrieval_query"
    genai.embed_content(**kwargs)


def _embed_content(text: str, task_type: Optional[str] = None) -> List[float]:
    """Embed text using the resolved model."""
    ensure_genai_configured()
    model = resolve_embed_model()
    kwargs: Dict = {"model": model, "content": text}

    # task_type is only guaranteed for embedding-001 (per Gemini SDK)
    if task_type and "embedding-001" in model:
        kwargs["task_type"] = task_type
        return genai.embed_content(**kwargs)["embedding"]

    if task_type and "gemini-embedding" in model:
        try:
            return genai.embed_content(**kwargs, task_type=task_type)["embedding"]
        except Exception:
            pass

    return genai.embed_content(**kwargs)["embedding"]

TRIAGE_SYSTEM = """You are an expert hospital triage nurse AI.
You MUST ground every routing decision in the RETRIEVED CLINICAL KNOWLEDGE provided.
Use general medical knowledge only to interpret symptom wording, never to override retrieved protocols."""


# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE  (loaded from JSON file)
# ─────────────────────────────────────────────

def load_knowledge_base(path: str = "knowledge_base.json") -> List[Dict]:
    """Load clinical documents, sanitize, and attach retrieval keywords."""
    with open(path, "r") as f:
        docs = json.load(f)
    print("[RAG] Sanitizing knowledge base (prompt-injection scan) …")
    for i, doc in enumerate(docs):
        enrich_document(doc)
        sanitize_document(doc)
        doc["_index"] = i
    return docs


def enrich_document(doc: Dict) -> Dict:
    """Add keywords and room/floor for hybrid retrieval if not already in the KB file."""
    doc_id = doc.get("id", "")
    aliases = SYMPTOM_ALIASES.get(doc_id, [])
    title_words = re.findall(r"[a-z0-9]+", doc.get("title", "").lower())
    doc["keywords"] = list(dict.fromkeys(aliases + title_words))

    loc = get_department_location(doc.get("department", ""))
    doc.setdefault("floor", loc["floor"])
    doc.setdefault("room", loc["room"])
    return doc


def build_index_text(doc: Dict) -> str:
    """
    Rich text for embedding — department, priority, keywords, and clinical content
    so retrieval matches how patients describe symptoms.
    """
    keywords = ", ".join(doc.get("keywords", []))
    priority_label = PRIORITY_LABELS.get(doc["priority"], "Unknown")
    return (
        f"Protocol: {doc['title']}\n"
        f"Department: {doc['department']}\n"
        f"Location: Floor {doc.get('floor', '1')}, Room {doc.get('room', '101')}\n"
        f"Triage Priority: {doc['priority']} ({priority_label})\n"
        f"Symptom keywords: {keywords}\n"
        f"Clinical guidance: {doc.get('content_safe', doc['content'])}"
    )


# ─────────────────────────────────────────────
# 2. STEP 1 — EMBED  (Lab 5: Embeddings)
# ─────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    """Convert patient symptoms into a query embedding vector."""
    return _embed_content(text, task_type="retrieval_query")


def embed_document(doc_text: str) -> List[float]:
    """Convert a knowledge-base document into an index embedding vector."""
    return _embed_content(doc_text, task_type="retrieval_document")


def build_vector_index(docs: List[Dict]) -> List[Dict]:
    """
    Pre-compute and attach an embedding vector to every knowledge-base document.
    In production this would be stored in a vector DB (e.g. Pinecone, Vertex AI
    Vector Search). Here we keep it in-memory for simplicity.
    """
    print(f"[RAG] Building vector index for {len(docs)} documents …")
    resolve_embed_model()
    print(f"[RAG] Config: TOP_K={TOP_K}, MIN_SIMILARITY={MIN_SIMILARITY}, KEYWORD_BOOST={KEYWORD_BOOST}")
    for doc in docs:
        doc["embedding"] = embed_document(build_index_text(doc))
        print(f"  [ok] Embedded: {doc['title']}")
    return docs


# ─────────────────────────────────────────────
# 3. STEP 2 — SEMANTIC SEARCH  (Lab 6: Retrieval)
# ─────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Measures how 'close' two vectors are in meaning.
    Returns a value in [-1, 1]; higher = more similar.
    (Lab 5 / Lab 6: why cosine similarity beats Euclidean for text)
    """
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _keyword_overlap_score(query: str, doc: Dict) -> float:
    """Hybrid boost: count matching symptom keywords in the patient query."""
    query_lower = query.lower()
    keywords = doc.get("keywords", [])
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in query_lower)
    return min(1.0, hits / max(3, len(keywords) * 0.25))


def _vector_search_candidates(
    query: str,
    query_embedding: List[float],
    indexed_docs: List[Dict],
    pool_size: int = RETRIEVE_CANDIDATES,
) -> List[Tuple[float, Dict, float]]:
    """First pass: semantic search + keyword boost, return candidate pool."""
    scored: List[Tuple[float, Dict, float, float]] = []
    for doc in indexed_docs:
        semantic = cosine_similarity(query_embedding, doc["embedding"])
        keyword = _keyword_overlap_score(query, doc)
        combined = semantic + KEYWORD_BOOST * keyword
        scored.append((combined, doc, semantic, keyword))

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [(c, d, s) for c, d, s, _ in scored if s >= MIN_SIMILARITY]

    if not filtered:
        best = max(scored, key=lambda x: x[2])
        filtered = [(best[0], best[1], best[2])]

    return filtered[:pool_size]


def retrieve(
    query: str,
    query_embedding: List[float],
    indexed_docs: List[Dict],
    reranker: HybridReranker,
    top_k: int = TOP_K,
) -> List[Tuple[float, Dict]]:
    """
    Two-stage retrieval:
      1. Vector search → candidate pool
      2. Hybrid rerank (semantic + TF-IDF + term overlap) → final top-k
    """
    pool = _vector_search_candidates(query, query_embedding, indexed_docs)
    candidates = [(sem, doc, doc["_index"]) for _, doc, sem in pool]
    reranked = reranker.rerank(query, candidates, top_k)

    results: List[Tuple[float, Dict]] = []
    for score, doc, breakdown in reranked:
        doc["_rerank_breakdown"] = breakdown
        results.append((score, doc))
    return results


def retrieve_debug(
    query: str,
    query_embedding: List[float],
    indexed_docs: List[Dict],
    reranker: HybridReranker,
    top_k: int = TOP_K,
) -> List[Dict]:
    """Detailed retrieval + rerank scores for tuning."""
    pool = _vector_search_candidates(query, query_embedding, indexed_docs)
    candidates = [(sem, doc, doc["_index"]) for _, doc, sem in pool]
    reranked = reranker.rerank(query, candidates, top_k)

    return [
        {
            "id": doc["id"],
            "title": doc["title"],
            "department": doc["department"],
            "priority": doc["priority"],
            "rerank_score": bd["rerank_score"],
            "semantic_score": bd["semantic"],
            "tfidf_score": bd["tfidf"],
            "term_overlap": bd["term_overlap"],
            "sanitization_flags": doc.get("sanitization_flags", []),
        }
        for _, doc, bd in reranked
    ]


# ─────────────────────────────────────────────
# 4. STEP 3 — CONTEXT INJECTION  (Lab 6: Augmentation)
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
Example 1 — Input: crushing chest pain, left arm numbness, sweating
→ department: "Cardiology Emergency", priority: 1, confidence: high

Example 2 — Input: 3-year-old child, fever 104°F, difficulty breathing
→ department: "Pediatrics Emergency", priority: 1, confidence: high

Example 3 — Input: small cut on finger, minimal bleeding, otherwise well
→ department: "Minor Injuries Unit", priority: 4, confidence: high
"""


def build_prompt(symptoms: str, retrieved: List[Tuple[float, Dict]]) -> str:
    """Augmented prompt with sanitized sources, tie-break rules, and few-shot examples."""
    if not retrieved:
        retrieved = []

    safe_symptoms, symptom_flags = sanitize_text(symptoms)
    if symptom_flags:
        print(f"[RAG] [WARN] Sanitized patient input flags: {', '.join(symptom_flags)}")

    top_score, top_doc = retrieved[0] if retrieved else (0.0, None)
    context_blocks = []
    for rank, (score, doc) in enumerate(retrieved, start=1):
        plabel = PRIORITY_LABELS.get(doc["priority"], "Unknown")
        bd = doc.get("_rerank_breakdown", {})
        context_blocks.append(
            f"[Rank {rank} | ID: {doc['id']} | Rerank: {score:.3f} | "
            f"sem={bd.get('semantic', 0):.2f} tfidf={bd.get('tfidf', 0):.2f} overlap={bd.get('term_overlap', 0):.2f}]\n"
            f"Department: {doc['department']}\n"
            f"Location: Floor {doc.get('floor', '1')}, Room {doc.get('room', '101')}\n"
            f"Required Priority: {doc['priority']} ({plabel})\n"
            f"Protocol: {doc.get('title_safe', doc['title'])}\n"
            f"{doc.get('content_safe', doc['content'])}"
        )
    context_str = "\n\n".join(context_blocks) if context_blocks else "(No matching protocols above threshold.)"

    grounding_rule = ""
    if top_doc and top_score >= STRONG_MATCH_THRESHOLD:
        plabel = PRIORITY_LABELS.get(top_doc["priority"], "Unknown")
        grounding_rule = (
            f"\nSTRONG MATCH (score {top_score:.2f}): The top protocol [{top_doc['id']}] "
            f"closely matches. You SHOULD route to department \"{top_doc['department']}\" "
            f"with priority {top_doc['priority']} ({plabel}) unless symptoms clearly indicate a different emergency.\n"
        )

    return f"""{FEW_SHOT_EXAMPLES}

──────────────────────────────
RETRIEVED CLINICAL KNOWLEDGE (ranked by relevance):
──────────────────────────────
{context_str}
{grounding_rule}
──────────────────────────────
PATIENT SYMPTOMS:
──────────────────────────────
{safe_symptoms}

──────────────────────────────
DECISION RULES:
──────────────────────────────
1. Prefer the highest-ranked protocol whose symptoms match the patient description.
2. department MUST be copied exactly from the chosen protocol's Department field.
3. priority MUST be an integer 1–4 matching the chosen protocol unless red-flag escalation is documented in the text.
4. priority_label MUST be: Critical (1), Urgent (2), Semi-urgent (3), Non-urgent (4).
5. confidence: "high" if top score ≥ {STRONG_MATCH_THRESHOLD:.2f}, "medium" if ≥ {MIN_SIMILARITY:.2f}, else "low".
6. If no protocol fits (all scores weak), use department "General Emergency" and priority 3 with confidence "low".
7. rationale: 2–3 sentences citing which protocol ID you followed and why.

Respond in this exact JSON format:
{{
  "department": "<exact department from protocol>",
  "priority": <1-4>,
  "priority_label": "<Critical | Urgent | Semi-urgent | Non-urgent>",
  "rationale": "<brief explanation referencing protocol ID>",
  "confidence": "<high | medium | low>",
  "matched_protocol_id": "<KB### or null>"
}}"""


def align_result_with_retrieval(
    result: Dict,
    retrieved: List[Tuple[float, Dict]],
) -> Dict:
    """
    Post-generation guardrail: when retrieval is very confident, ensure
    department/priority match the top knowledge-base protocol.
    """
    if not retrieved:
        return result

    top_score, top_doc = retrieved[0]
    if top_score < STRONG_MATCH_THRESHOLD:
        return result

    expected_label = PRIORITY_LABELS.get(top_doc["priority"], result.get("priority_label", ""))
    result["department"] = top_doc["department"]
    result["priority"] = top_doc["priority"]
    result["priority_label"] = expected_label
    result["matched_protocol_id"] = top_doc["id"]
    if result.get("confidence") == "low":
        result["confidence"] = "high"
    loc = get_department_location(result.get("department", top_doc["department"]))
    result["floor"] = top_doc.get("floor", loc["floor"])
    result["room"] = top_doc.get("room", loc["room"])
    return result


def attach_location(result: Dict, top_docs: List[Tuple[float, Dict]]) -> Dict:
    """Set room/floor from matched protocol or department lookup."""
    dept = result.get("department", "General Emergency")
    if top_docs:
        _, doc = top_docs[0]
        dept = result.get("department") or doc.get("department", dept)
        result["floor"] = doc.get("floor") or get_department_location(dept)["floor"]
        result["room"] = doc.get("room") or get_department_location(dept)["room"]
    else:
        loc = get_department_location(dept)
        result["floor"] = loc["floor"]
        result["room"] = loc["room"]
    return result


# ─────────────────────────────────────────────
# 5. STEP 4 — GENERATE  (Lab 7: Gemini API)
# ─────────────────────────────────────────────

INTAKE_SYSTEM = """You are a warm, professional healthcare intake assistant at a hospital.
Conduct a patient intake interview through natural conversation — like a nurse at the front desk.

Rules:
- Ask exactly ONE clear question per response (never a list of questions).
- Be empathetic and use simple, non-technical language.
- Gather information in this order (skip items already answered in the conversation):
  1. Patient full name
  2. Age and gender
  3. Chief complaint — what brings them in today
  4. When symptoms started and how they have changed
  5. Severity and related symptoms (only what is relevant to their complaint)
  6. Known allergies (or confirm none)
  7. Current medications and relevant medical history
- Do NOT diagnose, prescribe, or assign triage priority.
- Set intake_complete to true only when you have: name, chief complaint, symptom details with timing, and allergy status.
- When intake_complete is true, thank the patient and say you are passing their information to the clinical team.

Respond with valid JSON only (no markdown):
{
  "reply": "your next message to the patient",
  "intake_complete": false,
  "collected": {
    "name": null,
    "age": null,
    "gender": null,
    "chief_complaint": null,
    "symptoms": null,
    "onset": null,
    "allergies": null,
    "medications": null,
    "medical_history": null
  },
  "symptoms_summary": ""
}

When intake_complete is true, symptoms_summary must be a detailed plain-English paragraph covering everything collected, suitable for a triage nurse."""


def intake_chat(messages: List[Dict[str, str]]) -> Dict:
    """
    Multi-turn patient intake conversation. Returns the assistant's next
    question or signals intake_complete with a symptoms_summary for triage.
    """
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=INTAKE_SYSTEM)

    history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    last_user = messages[-1]["content"] if messages else ""
    if not last_user:
        last_user = "Begin the intake interview. Greet the patient and ask for their full name."

    chat = model.start_chat(history=history)
    response = chat.send_message(
        last_user,
        generation_config=genai.GenerationConfig(
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )
    result = json.loads(response.text)
    result.setdefault("intake_complete", False)
    result.setdefault("collected", {})
    result.setdefault("symptoms_summary", "")
    return result


def generate_triage(prompt: str) -> Dict:
    """Send the augmented prompt to Gemini and parse structured JSON."""
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=TRIAGE_SYSTEM)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=GENERATION_TEMPERATURE,
            response_mime_type="application/json",
        ),
    )
    result = json.loads(response.text)
    p = result.get("priority")
    if p in PRIORITY_LABELS:
        result["priority_label"] = PRIORITY_LABELS[p]
    return result


# ─────────────────────────────────────────────
# 6. FULL PIPELINE — ORCHESTRATION
# ─────────────────────────────────────────────

class TriageRAGPipeline:
    """
    Wraps the full embed → search → inject → respond flow
    into a single reusable class for the Cloud Run backend.
    """

    def __init__(self, knowledge_base_path: str = "knowledge_base.json"):
        ensure_genai_configured()
        docs = load_knowledge_base(knowledge_base_path)
        self.indexed_docs = build_vector_index(docs)
        self.reranker = HybridReranker(self.indexed_docs)
        print("[RAG] Hybrid reranker ready (semantic + TF-IDF + term overlap).\n")
        print("[RAG] Pipeline ready.\n")

    def triage(self, symptoms: str) -> Dict:
        """
        End-to-end triage for a patient's symptom description.

        Returns a dict with: department, priority, priority_label,
                             rationale, confidence, retrieved_sources
        """
        # ── Step 1: Embed the incoming symptoms ──────────────────
        print(f"[RAG] Step 1 — Embedding query: '{symptoms[:60]}…'")
        query_vec = embed_text(symptoms)

        # ── Step 2: Semantic search over knowledge base ───────────
        print("[RAG] Step 2 — Semantic search …")
        top_docs = retrieve(symptoms, query_vec, self.indexed_docs, self.reranker, top_k=TOP_K)
        for score, doc in top_docs:
            print(f"  → [{score:.3f}] {doc['id']} {doc['title']}")

        # ── Step 3: Inject context into prompt ────────────────────
        print("[RAG] Step 3 — Building augmented prompt …")
        prompt = build_prompt(symptoms, top_docs)

        # ── Step 4: Generate + align with strong retrieval match ──
        print("[RAG] Step 4 — Calling Gemini …")
        result = generate_triage(prompt)
        result = align_result_with_retrieval(result, top_docs)
        result = attach_location(result, top_docs)

        top_score = top_docs[0][0] if top_docs else 0.0
        result["retrieval_top_score"] = round(top_score, 3)
        result["retrieved_sources"] = []
        for score, doc in top_docs:
            bd = doc.get("_rerank_breakdown", {})
            result["retrieved_sources"].append({
                "title": doc["title"],
                "id": doc["id"],
                "score": round(score, 3),
                "department": doc["department"],
                "priority": doc["priority"],
                "semantic_score": bd.get("semantic"),
                "tfidf_score": bd.get("tfidf"),
                "term_overlap": bd.get("term_overlap"),
                "sanitized": bool(doc.get("sanitization_flags")),
            })

        print(f"[RAG] Done - {result['department']} | P{result['priority']} | matched {result.get('matched_protocol_id')}\n")
        return result

    def retrieve_only(self, symptoms: str) -> Dict:
        """Debug retrieval without calling Gemini generation."""
        query_vec = embed_text(symptoms)
        ranked = retrieve_debug(symptoms, query_vec, self.indexed_docs, self.reranker, top_k=TOP_K)
        return {
            "symptoms": symptoms,
            "config": {
                "top_k": TOP_K,
                "retrieve_candidates": RETRIEVE_CANDIDATES,
                "min_similarity": MIN_SIMILARITY,
                "keyword_boost": KEYWORD_BOOST,
                "strong_match": STRONG_MATCH_THRESHOLD,
                "rerank_weights": {
                    "semantic": RERANK_WEIGHT_SEMANTIC,
                    "tfidf": RERANK_WEIGHT_TFIDF,
                    "overlap": RERANK_WEIGHT_OVERLAP,
                },
            },
            "results": ranked,
        }


# ─────────────────────────────────────────────
# 7. QUICK LOCAL TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = TriageRAGPipeline("knowledge_base.json")

    test_cases = [
        "Patient has crushing chest pain, sweating, and pain radiating to the left arm.",
        "Child aged 3, fever of 104.5°F for two days, difficulty breathing.",
        "Patient has a small cut on the finger, no bleeding, no other symptoms.",
    ]

    for symptoms in test_cases:
        print("=" * 60)
        print(f"SYMPTOMS: {symptoms}")
        print("=" * 60)
        result = pipeline.triage(symptoms)
        print(json.dumps(result, indent=2))
        print()
