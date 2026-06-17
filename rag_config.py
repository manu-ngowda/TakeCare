"""
RAG tuning parameters — override via environment variables.
"""

import os

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RETRIEVE_CANDIDATES = int(os.getenv("RAG_RETRIEVE_CANDIDATES", "10"))
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SCORE", "0.42"))
KEYWORD_BOOST = float(os.getenv("RAG_KEYWORD_BOOST", "0.08"))
STRONG_MATCH_THRESHOLD = float(os.getenv("RAG_STRONG_MATCH", "0.72"))
GENERATION_TEMPERATURE = float(os.getenv("RAG_GEN_TEMP", "0.1"))

# Lab 8 — Hybrid rerank weights (must sum to ~1.0)
RERANK_WEIGHT_SEMANTIC = float(os.getenv("RAG_RERANK_SEMANTIC", "0.50"))
RERANK_WEIGHT_TFIDF = float(os.getenv("RAG_RERANK_TFIDF", "0.35"))
RERANK_WEIGHT_OVERLAP = float(os.getenv("RAG_RERANK_OVERLAP", "0.15"))

PRIORITY_LABELS = {
    1: "Critical",
    2: "Urgent",
    3: "Semi-urgent",
    4: "Non-urgent",
}

# Room and floor for each department (shown to patients after triage)
DEPARTMENT_LOCATIONS: dict[str, dict[str, str]] = {
    "Cardiology Emergency": {"floor": "2", "room": "204"},
    "Pediatrics Emergency": {"floor": "3", "room": "301"},
    "Neurology Emergency": {"floor": "4", "room": "402"},
    "Emergency Medicine": {"floor": "1", "room": "118"},
    "Pulmonology Emergency": {"floor": "2", "room": "215"},
    "General Surgery Emergency": {"floor": "1", "room": "112"},
    "Cardiology": {"floor": "2", "room": "208"},
    "Endocrinology Emergency": {"floor": "2", "room": "212"},
    "Neurosurgery Emergency": {"floor": "4", "room": "410"},
    "Burns Unit": {"floor": "1", "room": "105"},
    "Orthopaedics": {"floor": "3", "room": "318"},
    "Urology / General Medicine": {"floor": "3", "room": "305"},
    "Psychiatry Emergency": {"floor": "5", "room": "501"},
    "Ophthalmology Emergency": {"floor": "2", "room": "220"},
    "ENT Outpatient": {"floor": "3", "room": "312"},
    "Minor Injuries Unit": {"floor": "1", "room": "102"},
    "Obstetrics Emergency": {"floor": "2", "room": "230"},
    "Toxicology / Emergency Medicine": {"floor": "1", "room": "120"},
    "General Outpatient / Physiotherapy": {"floor": "1", "room": "110"},
    "Dermatology / Infectious Disease": {"floor": "3", "room": "308"},
    "General Medicine": {"floor": "3", "room": "302"},
    "Dental Emergency": {"floor": "2", "room": "225"},
    "General Outpatient": {"floor": "1", "room": "108"},
    "General Emergency": {"floor": "1", "room": "101"},
}


def get_department_location(department: str) -> dict[str, str]:
    return DEPARTMENT_LOCATIONS.get(
        department,
        DEPARTMENT_LOCATIONS["General Emergency"],
    )

# Symptom aliases improve hybrid (keyword) retrieval
SYMPTOM_ALIASES: dict[str, list[str]] = {
    "KB001": ["chest pain", "heart attack", "myocardial infarction", "left arm pain", "arm numbness", "jaw pain", "sweating", "diaphoresis", "shortness of breath", "dyspnea"],
    "KB002": ["child fever", "pediatric fever", "high fever", "104 fever", "febrile", "infant fever"],
    "KB003": ["stroke", "face drooping", "arm weakness", "slurred speech", "fast", "sudden weakness"],
    "KB004": ["anaphylaxis", "allergic reaction", "throat swelling", "hives", "epipen", "difficulty breathing after food"],
    "KB005": ["shortness of breath", "can't breathe", "respiratory distress", "wheezing", "asthma attack", "low oxygen"],
    "KB006": ["abdominal pain", "stomach pain", "appendicitis", "severe belly pain"],
    "KB007": ["high blood pressure", "hypertensive", "bp 180", "severe headache bp"],
    "KB008": ["hypoglycemia", "low blood sugar", "diabetic unconscious", "sweating confusion diabetes"],
    "KB009": ["head injury", "head trauma", "fall hit head", "concussion"],
    "KB010": ["burn", "burns", "scald", "fire injury"],
    "KB011": ["fracture", "broken bone", "limb deformity", "fall leg", "knee injury"],
    "KB012": ["uti", "urinary infection", "flank pain fever", "painful urination"],
    "KB013": ["suicidal", "self harm", "want to die", "mental health crisis"],
    "KB014": ["chemical in eye", "eye splash", "eye burn"],
    "KB015": ["ear pain", "ear infection", "otitis"],
    "KB016": ["small cut", "minor cut", "finger cut", "superficial wound", "laceration minor"],
    "KB017": ["pregnant", "labour", "contractions", "pregnancy bleeding"],
    "KB018": ["overdose", "drug overdose", "poisoning", "took too many pills"],
    "KB019": ["back pain chronic", "lower back pain long term"],
    "KB020": ["rash fever", "skin rash", "petechial rash"],
    "KB021": ["seizure", "convulsion", "fitting", "epilepsy attack"],
    "KB022": ["vomiting diarrhea", "gastroenteritis", "dehydration vomiting"],
    "KB023": ["toothache", "dental pain", "tooth abscess"],
    "KB024": ["palpitations", "irregular heartbeat", "heart racing dizzy"],
    "KB025": ["vaccination", "routine checkup", "health check", "prescription refill"],
}
