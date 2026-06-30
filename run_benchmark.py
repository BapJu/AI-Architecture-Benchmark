"""
run_benchmark.py — Entry point

Usage:
    python run_benchmark.py

Requires .env with:
    GOOGLE_API_KEY=your_key_here

Optional env vars (set before running, PowerShell syntax):
    $env:BENCHMARK_QUERIES="Q01,Q03,Q07"   # Run a subset of queries
    $env:RATE_LIMIT_PAUSE="2.0"             # Seconds between API calls (default 1.5)
    $env:OUTPUT_DIR="results"               # Where to save CSV (default: results/)
"""

import json
import os
import sys
from pathlib import Path

# ── Ensure project root is on sys.path so 'benchmark' package is importable ──
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Load environment ──────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not set. Add it to .env or your environment.")
    print("  echo 'GOOGLE_API_KEY=your_key_here' > .env")
    sys.exit(1)

# ── Imports ───────────────────────────────────────────────────────────────────
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

from benchmark.src.approach_b.vector_store import build_vector_store
from benchmark.src.eval.queries import QUERIES
from benchmark.src.eval.harness import run_full_benchmark, save_results, print_summary_table

# ── Config ────────────────────────────────────────────────────────────────────
PROFILES_PATH = Path("benchmark/data/profiles.json")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "results"))
RATE_LIMIT_PAUSE = float(os.getenv("RATE_LIMIT_PAUSE", "1.5"))

# Allow running a subset of queries
query_filter = os.getenv("BENCHMARK_QUERIES", "")
if query_filter:
    selected_ids = {q.strip() for q in query_filter.split(",")}
    active_queries = [q for q in QUERIES if q["id"] in selected_ids]
else:
    active_queries = QUERIES


def main():
    print("=" * 60)
    print("  LinkedIn Lead Gen — AI Architecture Benchmark")
    print("  Helloo.in (Naive) vs LangGraph ReAct Agent (RAG)")
    print("=" * 60)

    # ── Load profiles ─────────────────────────────────────────────────────────
    print(f"\nLoading profiles from {PROFILES_PATH}...")
    with open(PROFILES_PATH, encoding="utf-8") as f:
        profiles = json.load(f)
    profiles_by_id = {p["id"]: p for p in profiles}
    print(f"  {len(profiles)} profiles loaded.")

    # ── Init Gemini clients ───────────────────────────────────────────────────
    print("\nInitializing Gemini clients...")
    genai_client = genai.Client(api_key=API_KEY)

    # LangChain wrapper for the agent
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=API_KEY,
        temperature=0.1,
    )

    # ── Build vector store ────────────────────────────────────────────────────
    print("Building Chroma vector store (Gemini gemini-embedding-001)...")
    collection = build_vector_store(profiles, genai_client)
    print(f"  Vector store ready: {len(profiles) * 3} chunks indexed.")

    # ── Run benchmark ─────────────────────────────────────────────────────────
    results = run_full_benchmark(
        client=genai_client,
        llm=llm,
        profiles=profiles,
        collection=collection,
        profiles_by_id=profiles_by_id,
        queries=active_queries,
        output_dir=OUTPUT_DIR,
        rate_limit_pause=RATE_LIMIT_PAUSE,
    )

    # ── Save & display ────────────────────────────────────────────────────────
    csv_path = save_results(results, OUTPUT_DIR)
    print_summary_table(results)
    print(f"Full results saved to: {csv_path}\n")


if __name__ == "__main__":
    main()
