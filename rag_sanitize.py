"""
rag_sanitize.py — Lab 8 Module 3: Document sanitization layer.

Scans knowledge-base text before it is injected into Gemini prompts.
Flags and strips common prompt-injection patterns.
"""

import re
from typing import List, Tuple

# Case-insensitive patterns that must never reach the LLM as instructions
INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("ignore_previous_instructions", re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        re.I,
    )),
    ("disregard_context", re.compile(
        r"disregard\s+(all\s+)?(previous|prior|above|the)\s+(context|instructions?|rules?)",
        re.I,
    )),
    ("reveal_confidential", re.compile(
        r"reveal\s+(all\s+)?(confidential|secret|private|system)\s+(data|information|prompts?|keys?)",
        re.I,
    )),
    ("override_instructions", re.compile(
        r"override\s+(the\s+)?(system|safety|triage)\s+(instructions?|rules?|prompt)",
        re.I,
    )),
    ("you_are_now", re.compile(
        r"you\s+are\s+now\s+(a|an|the)\s+",
        re.I,
    )),
    ("new_instructions", re.compile(
        r"(your\s+)?new\s+instructions?\s+(are|is|:)",
        re.I,
    )),
    ("act_as_unrestricted", re.compile(
        r"act\s+as\s+(an?\s+)?(unrestricted|unfiltered|jailbroken)\s+",
        re.I,
    )),
    ("system_prompt_leak", re.compile(
        r"(print|show|output|repeat)\s+(the\s+)?(system|hidden|original)\s+prompt",
        re.I,
    )),
    ("roleplay_escape", re.compile(
        r"pretend\s+(you\s+)?(are|have)\s+no\s+(rules|restrictions|guidelines)",
        re.I,
    )),
]

REPLACEMENT = "[REMOVED: potential prompt injection]"


def sanitize_text(text: str) -> Tuple[str, List[str]]:
    """
    Scan text for injection patterns. Returns cleaned text and list of flags.
    """
    if not text:
        return text, []

    flags: List[str] = []
    cleaned = text

    for flag_name, pattern in INJECTION_PATTERNS:
        if pattern.search(cleaned):
            flags.append(flag_name)
            cleaned = pattern.sub(REPLACEMENT, cleaned)

    # Collapse repeated replacement markers
    cleaned = re.sub(
        rf"(?:{re.escape(REPLACEMENT)}\s*){{2,}}",
        REPLACEMENT + " ",
        cleaned,
    ).strip()

    return cleaned, flags


def sanitize_document(doc: dict) -> dict:
    """Sanitize title and content fields on a knowledge-base document."""
    safe_title, title_flags = sanitize_text(doc.get("title", ""))
    safe_content, content_flags = sanitize_text(doc.get("content", ""))

    doc["title_safe"] = safe_title
    doc["content_safe"] = safe_content
    doc["sanitization_flags"] = title_flags + content_flags
    if doc["sanitization_flags"]:
        print(f"  [WARN] Sanitized {doc.get('id', '?')}: {', '.join(doc['sanitization_flags'])}")
    return doc
