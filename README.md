# LinkedIn Lead Gen — AI Architecture Benchmark

## Context

This project benchmarks two AI architectures for the same task: **querying a corpus of LinkedIn profiles** to support B2B lead generation.

- **Approach A (Naive)** — mirrors the production pattern from [Helloo.in](https://helloo.in), a real Django SaaS I built: full JSON context stuffing into a single prompt.
- **Approach B (Agentic RAG)** — the modern architecture: a LangGraph ReAct agent backed by a Chroma vector store.

Both use **Gemini 2.0 Flash** and identical fake profile data. No real LinkedIn API is needed.

---

## Project Structure

```
benchmark/
├── data/
│   └── profiles.json          # 25 synthetic LinkedIn profiles
├── src/
│   ├── approach_a/
│   │   └── naive.py           # Single-shot full-context prompt (Helloo.in pattern)
│   ├── approach_b/
│   │   ├── vector_store.py    # Chroma + Gemini text-embedding-004
│   │   └── agent.py           # LangGraph ReAct agent with search_profiles tool
│   └── eval/
│       ├── queries.py         # 10 benchmark queries with ground truth
│       └── harness.py         # Run both approaches, score with LLM-as-judge, export CSV
├── run_benchmark.py           # Entry point
└── .env                       # GOOGLE_API_KEY=...
```

---

## Setup

```bash
# Install deps
pip install google-genai chromadb langgraph langchain-google-genai langchain-core pandas tabulate python-dotenv

# Add your key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run
python run_benchmark.py
```

---

## What It Measures

| Metric | Why It Matters |
|---|---|
| Input tokens | Direct cost proxy |
| Latency (s) | User-facing responsiveness |
| Relevance score (1–5) | Did the answer address the question? |
| Fact score (0–100%) | Were stated facts verifiable in the data? |
| Tool calls | Reasoning transparency |

---

## Key Insight

```
================================================================================
  BENCHMARK RESULTS — Per Query
================================================================================
╭──────┬─────────────┬────────┬────────────┬────────────┬───────────┬────────────┬────────────┬───────────┬───────────┬───────────┬───────────╮
│ ID   │ Category    │ Diff   │   A:Tokens │   B:Tokens │ Token Δ   │   A:Lat(s) │   B:Lat(s) │ A:Relev   │ B:Relev   │ A:Facts   │ B:Facts   │
├──────┼─────────────┼────────┼────────────┼────────────┼───────────┼────────────┼────────────┼───────────┼───────────┼───────────┼───────────┤
│ Q01  │ lookup      │ simple │       8817 │       1268 │ +85.6%    │      0.681 │      1.869 │ 5/5       │ 5/5       │ 100%      │ 100%      │
│ Q02  │ lookup      │ simple │       8838 │       1510 │ +82.9%    │      0.795 │      2.908 │ 5/5       │ 5/5       │ 75%       │ 100%      │
│ Q03  │ filter      │ medium │       8864 │       1992 │ +77.5%    │      0.894 │      2.889 │ 4/5       │ 5/5       │ 100%      │ 100%      │
│ Q04  │ filter      │ medium │       8874 │       1319 │ +85.1%    │      1.147 │      2.513 │ 5/5       │ 4/5       │ 100%      │ 75%       │
│ Q05  │ semantic    │ medium │       8936 │       2014 │ +77.5%    │      1.195 │      5.039 │ 5/5       │ 5/5       │ 100%      │ 100%      │
│ Q06  │ semantic    │ medium │       8879 │       1674 │ +81.1%    │      0.972 │      3.107 │ 4/5       │ 4/5       │ 67%       │ 67%       │
│ Q07  │ synthesis   │ hard   │       9332 │       3541 │ +62.1%    │      3.91  │      9.65  │ 5/5       │ 5/5       │ 100%      │ 100%      │
│ Q08  │ synthesis   │ hard   │       9325 │       1946 │ +79.1%    │      3.326 │      5.024 │ 5/5       │ 4/5       │ 100%      │ 80%       │
│ Q09  │ aggregation │ medium │       8835 │       2218 │ +74.9%    │      0.666 │      4.024 │ 5/5       │ 5/5       │ 100%      │ 100%      │
│ Q10  │ aggregation │ hard   │       9319 │       6023 │ +35.4%    │      3.199 │     12.81  │ 0/5       │ 0/5       │ 0%        │ 0%        │
╰──────┴─────────────┴────────┴────────────┴────────────┴───────────┴────────────┴────────────┴───────────┴───────────┴───────────┴───────────╯

================================================================================
  AGGREGATE SUMMARY
================================================================================
╭─────────────────────┬──────────────────────┬──────────────────────┬─────────╮
│ Metric              │   Approach A (Naive) │   Approach B (Agent) │ Delta   │
├─────────────────────┼──────────────────────┼──────────────────────┼─────────┤
│ Avg Input Tokens    │              8809.5  │              1588.8  │ -7221   │
│ Avg Total Tokens    │              9001.9  │              2350.5  │ -6651   │
│ Avg Latency (s)     │                 1.68 │                 4.98 │ +3.300  │
│ Avg Relevance Score │                 4.3  │                 4.2  │ -0.10   │
│ Avg Fact Score (%)  │                84.2  │                82.2  │ -2.0%   │
│ Avg Tool Calls      │                 0    │                 1.4  │ +1.4    │
╰─────────────────────┴──────────────────────┴──────────────────────┴─────────╯
```

Full results saved to: results\benchmark_20260630_195244.csv


## Why did tokens drop by 74%?
Approach A (Naive) forces the LLM to read the entire database of LinkedIn profiles for every single question, resulting in ~8,800 input tokens per call. Approach B (Agentic RAG) uses a vector database (Chroma) to filter the data before the LLM reads it. By only passing the 1 or 2 relevant profiles to the LLM instead of all 25, we cut input token consumption by roughly 74% (saving ~6,600 total tokens per query) while maintaining nearly identical accuracy and relevance scores.

## The Trade-off (The Latency Tax)
Make sure to also point out the Latency column. While Approach B saves a ton of money on tokens, it takes about 3.3 seconds longer on average.

This is the classic RAG trade-off:

Approach A: Expensive (high tokens), but fast (single shot).

Approach B: Cheap (low tokens), but slow (requires a database search step + tool calling overhead).