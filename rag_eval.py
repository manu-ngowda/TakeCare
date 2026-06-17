"""
rag_eval.py — Measure retrieval and end-to-end triage accuracy.

Usage (PowerShell):
    $env:GEMINI_API_KEY = "your-key-here"
    python rag_eval.py --retrieval-only

Usage (CMD):
    set GEMINI_API_KEY=your-key-here
    python rag_eval.py
"""

import argparse
import json
import sys

from rag_pipeline import TriageRAGPipeline, ensure_genai_configured


def load_cases(path: str = "rag_eval.json"):
    with open(path) as f:
        return json.load(f)


def eval_retrieval(pipeline: TriageRAGPipeline, cases: list) -> dict:
    hits_top1 = hits_top3 = 0
    rows = []

    for case in cases:
        out = pipeline.retrieve_only(case["symptoms"])
        results = out["results"]
        top1_id = results[0]["id"] if results else None
        top3_ids = [r["id"] for r in results[:3]]
        expected = case.get("expected_protocol")

        ok1 = top1_id == expected
        ok3 = expected in top3_ids
        hits_top1 += int(ok1)
        hits_top3 += int(ok3)

        rows.append({
            "symptoms": case["symptoms"][:60] + "…",
            "expected": expected,
            "top1": top1_id,
            "top1_score": results[0]["semantic_score"] if results else 0,
            "top3": top3_ids,
            "top1_ok": ok1,
        })

    n = len(cases)
    return {
        "mode": "retrieval",
        "cases": n,
        "top1_accuracy": round(hits_top1 / n, 3) if n else 0,
        "top3_accuracy": round(hits_top3 / n, 3) if n else 0,
        "rows": rows,
    }


def eval_full(pipeline: TriageRAGPipeline, cases: list) -> dict:
    dept_ok = pri_ok = proto_ok = 0
    rows = []

    for case in cases:
        result = pipeline.triage(case["symptoms"])
        matched = result.get("matched_protocol_id")
        sources = result.get("retrieved_sources", [])
        top1 = sources[0]["id"] if sources else None

        d_ok = result["department"] == case["expected_department"]
        p_ok = result["priority"] == case["expected_priority"]
        m_ok = matched == case.get("expected_protocol") or top1 == case.get("expected_protocol")

        dept_ok += int(d_ok)
        pri_ok += int(p_ok)
        proto_ok += int(m_ok)

        rows.append({
            "symptoms": case["symptoms"][:50] + "…",
            "expected": f"{case['expected_department']} P{case['expected_priority']}",
            "got": f"{result['department']} P{result['priority']}",
            "matched_protocol": matched,
            "top_retrieval": top1,
            "dept_ok": d_ok,
            "priority_ok": p_ok,
            "protocol_ok": m_ok,
        })

    n = len(cases)
    return {
        "mode": "full",
        "cases": n,
        "department_accuracy": round(dept_ok / n, 3) if n else 0,
        "priority_accuracy": round(pri_ok / n, 3) if n else 0,
        "protocol_accuracy": round(proto_ok / n, 3) if n else 0,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG triage pipeline")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip Gemini generation")
    parser.add_argument("--cases", default="rag_eval.json", help="Path to eval cases JSON")
    args = parser.parse_args()

    try:
        ensure_genai_configured()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    cases = load_cases(args.cases)
    print(f"Loading pipeline and running {len(cases)} eval cases …\n")
    pipeline = TriageRAGPipeline()

    report = eval_retrieval(pipeline, cases) if args.retrieval_only else eval_full(pipeline, cases)
    print(json.dumps(report, indent=2))

    if report["mode"] == "retrieval":
        ok = report["top1_accuracy"] >= 0.7
        print(f"\nTop-1 retrieval: {report['top1_accuracy']*100:.0f}%  Top-3: {report['top3_accuracy']*100:.0f}%")
    else:
        ok = report["department_accuracy"] >= 0.7
        print(f"\nDept: {report['department_accuracy']*100:.0f}%  Priority: {report['priority_accuracy']*100:.0f}%  Protocol: {report['protocol_accuracy']*100:.0f}%")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
