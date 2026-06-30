"""
Evaluation Harness

Runs all 10 queries against both approaches, scores each answer with
an LLM-as-judge (Gemini), and outputs a CSV + formatted summary table.

LLM-as-judge metrics:
  - relevance_score: 1–5  (does the answer address the question?)
  - fact_score: 0–100     (are stated facts grounded in the profile data?)
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
from tabulate import tabulate

JUDGE_MODEL = "gemini-2.5-flash"

JUDGE_PROMPT = """You are an impartial evaluator for a B2B sales intelligence system.

You will be given:
- A user query
- The ground truth hints (key facts the answer should mention)
- A system answer to evaluate

Score the answer on two dimensions:

1. relevance_score (1–5):
   1 = Completely irrelevant or refused to answer
   2 = Partially addresses the question but misses key points
   3 = Addresses the question but with gaps or vagueness
   4 = Good answer, addresses the question well
   5 = Excellent, precise, complete answer

2. fact_score (0–100):
   Percentage of ground truth hints that appear (or are strongly implied) in the answer.
   Count how many hints are present / total hints * 100.

Return ONLY a JSON object like:
{{"relevance_score": 4, "fact_score": 75, "reasoning": "one sentence"}}

Query: {query}
Ground truth hints: {hints}
Answer to evaluate: {answer}"""


def judge_answer(
    client: genai.Client,
    query: str,
    hints: list[str],
    answer: str,
) -> dict:
    """Call Gemini to score an answer. Returns relevance_score, fact_score, reasoning."""
    prompt = JUDGE_PROMPT.format(
        query=query,
        hints=json.dumps(hints),
        answer=answer,
    )
    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        scores = json.loads(response.text)
        return {
            "relevance_score": int(scores.get("relevance_score", 0)),
            "fact_score": float(scores.get("fact_score", 0)),
            "judge_reasoning": scores.get("judge_reasoning", scores.get("reasoning", "")),
        }
    except Exception as e:
        return {"relevance_score": 0, "fact_score": 0.0, "judge_reasoning": f"Judge error: {e}"}


def run_full_benchmark(
    client: genai.Client,
    llm,
    profiles: list[dict],
    collection,
    profiles_by_id: dict,
    queries: list[dict],
    output_dir: Path,
    rate_limit_pause: float = 45.0,
) -> list[dict]:
    """
    Run all queries × both approaches, score, return results list.
    """
    from benchmark.src.approach_a.naive import run_naive_query
    from benchmark.src.approach_b.agent import run_agent_query

    all_results = []

    total = len(queries) * 2
    print(f"\n{'='*60}")
    print(f"  Running benchmark: {len(queries)} queries × 2 approaches = {total} calls")
    print(f"{'='*60}\n")

    for i, q_data in enumerate(queries):
        qid = q_data["id"]
        query = q_data["query"]
        hints = q_data["ground_truth_hints"]
        category = q_data["category"]
        difficulty = q_data["difficulty"]

        print(f"[{qid}] {category.upper()} / {difficulty} — {query[:60]}...")

        # ── Approach A ────────────────────────────────────────────────────────
        print("  → Approach A (naive)...", end="", flush=True)
        result_a = run_naive_query(client, profiles, query)
        print(f" {result_a['latency_s']}s | {result_a['total_tokens']} tokens")

        time.sleep(rate_limit_pause)

        # ── Approach B ────────────────────────────────────────────────────────
        print("  → Approach B (agent)...", end="", flush=True)
        result_b = run_agent_query(llm, collection, profiles_by_id, query)
        print(f" {result_b['latency_s']}s | {result_b['total_tokens']} tokens | {result_b['tool_calls']} tool calls")

        time.sleep(rate_limit_pause)

        # ── Judge both answers ─────────────────────────────────────────────
        print("  → Judging...", end="", flush=True)
        scores_a = judge_answer(client, query, hints, result_a["answer"])
        time.sleep(rate_limit_pause)
        scores_b = judge_answer(client, query, hints, result_b["answer"])
        time.sleep(rate_limit_pause)
        print(f" A={scores_a['relevance_score']}/5 ({scores_a['fact_score']}%) | B={scores_b['relevance_score']}/5 ({scores_b['fact_score']}%)")

        # ── Combine rows ──────────────────────────────────────────────────────
        for result, scores in [(result_a, scores_a), (result_b, scores_b)]:

            # Safely cast the answer to a string whether it's a list or not
            raw_answer = result["answer"]
            if isinstance(raw_answer, list):
                str_answer = "".join(str(item) for item in raw_answer)
            else:
                str_answer = str(raw_answer)

            all_results.append({
                "query_id": qid,
                "category": category,
                "difficulty": difficulty,
                "query": query,
                "approach": result["approach"],
                "answer_preview": str_answer[:200].replace("\n", " "),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"],
                "latency_s": result["latency_s"],
                "tool_calls": result["tool_calls"],
                "relevance_score": scores["relevance_score"],
                "fact_score": scores["fact_score"],
                "judge_reasoning": scores["judge_reasoning"],
            })

    return all_results


def save_results(results: list[dict], output_dir: Path) -> Path:
    """Save results to CSV and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_{timestamp}.csv"

    fieldnames = [
        "query_id", "category", "difficulty", "query", "approach",
        "input_tokens", "output_tokens", "total_tokens", "latency_s",
        "tool_calls", "relevance_score", "fact_score", "judge_reasoning",
        "answer_preview",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return csv_path


def print_summary_table(results: list[dict]):
    """Print a formatted comparison table to stdout."""

    def avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else 0

    rows_a = [r for r in results if r["approach"] == "A_naive"]
    rows_b = [r for r in results if r["approach"] == "B_agent"]

    # Per-query comparison
    query_rows = []
    query_ids = sorted({r["query_id"] for r in results})
    for qid in query_ids:
        a = next((r for r in rows_a if r["query_id"] == qid), None)
        b = next((r for r in rows_b if r["query_id"] == qid), None)
        if a and b:
            token_savings = round((1 - b["total_tokens"] / max(a["total_tokens"], 1)) * 100, 1)
            query_rows.append([
                qid,
                a["category"],
                a["difficulty"],
                a["total_tokens"],
                b["total_tokens"],
                f"{token_savings:+.1f}%",
                a["latency_s"],
                b["latency_s"],
                f"{a['relevance_score']}/5",
                f"{b['relevance_score']}/5",
                f"{a['fact_score']:.0f}%",
                f"{b['fact_score']:.0f}%",
            ])

    headers = [
        "ID", "Category", "Diff",
        "A:Tokens", "B:Tokens", "Token Δ",
        "A:Lat(s)", "B:Lat(s)",
        "A:Relev", "B:Relev",
        "A:Facts", "B:Facts",
    ]

    print("\n" + "="*80)
    print("  BENCHMARK RESULTS — Per Query")
    print("="*80)
    print(tabulate(query_rows, headers=headers, tablefmt="rounded_outline"))

    # Aggregate summary
    print("\n" + "="*80)
    print("  AGGREGATE SUMMARY")
    print("="*80)
    summary = [
        ["Metric", "Approach A (Naive)", "Approach B (Agent)", "Delta"],
        [
            "Avg Input Tokens",
            avg([r["input_tokens"] for r in rows_a]),
            avg([r["input_tokens"] for r in rows_b]),
            f"{avg([r['input_tokens'] for r in rows_b]) - avg([r['input_tokens'] for r in rows_a]):+.0f}",
        ],
        [
            "Avg Total Tokens",
            avg([r["total_tokens"] for r in rows_a]),
            avg([r["total_tokens"] for r in rows_b]),
            f"{avg([r['total_tokens'] for r in rows_b]) - avg([r['total_tokens'] for r in rows_a]):+.0f}",
        ],
        [
            "Avg Latency (s)",
            avg([r["latency_s"] for r in rows_a]),
            avg([r["latency_s"] for r in rows_b]),
            f"{avg([r['latency_s'] for r in rows_b]) - avg([r['latency_s'] for r in rows_a]):+.3f}",
        ],
        [
            "Avg Relevance Score",
            avg([r["relevance_score"] for r in rows_a]),
            avg([r["relevance_score"] for r in rows_b]),
            f"{avg([r['relevance_score'] for r in rows_b]) - avg([r['relevance_score'] for r in rows_a]):+.2f}",
        ],
        [
            "Avg Fact Score (%)",
            avg([r["fact_score"] for r in rows_a]),
            avg([r["fact_score"] for r in rows_b]),
            f"{avg([r['fact_score'] for r in rows_b]) - avg([r['fact_score'] for r in rows_a]):+.1f}%",
        ],
        [
            "Avg Tool Calls",
            avg([r["tool_calls"] for r in rows_a]),
            avg([r["tool_calls"] for r in rows_b]),
            f"{avg([r['tool_calls'] for r in rows_b]) - avg([r['tool_calls'] for r in rows_a]):+.1f}",
        ],
    ]
    print(tabulate(summary[1:], headers=summary[0], tablefmt="rounded_outline"))
    print()
