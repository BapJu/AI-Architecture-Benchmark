"""
Benchmark query suite — 10 queries covering 5 query categories.

Each query has:
  - id: unique identifier
  - category: type of reasoning required
  - query: the actual question
  - ground_truth_hints: key facts the answer MUST contain (used by LLM judge)
  - difficulty: simple / medium / hard
"""

QUERIES = [
    # ─── Category 1: Simple Lookup ────────────────────────────────────────────
    {
        "id": "Q01",
        "category": "lookup",
        "difficulty": "simple",
        "query": "What is Sophie Marchand's current role and company?",
        "ground_truth_hints": ["VP of Sales", "EMEA", "Salesforce"],
    },
    {
        "id": "Q02",
        "category": "lookup",
        "difficulty": "simple",
        "query": "Who works at Mistral AI and what is their specialty?",
        "ground_truth_hints": ["Yuki Tanaka", "AI Research", "LLM", "Mixtral"],
    },

    # ─── Category 2: Filtering ────────────────────────────────────────────────
    {
        "id": "Q03",
        "category": "filter",
        "difficulty": "medium",
        "query": "List all C-Suite executives in the FinTech industry.",
        "ground_truth_hints": ["Thomas Keller", "Priya Nair", "Fatima Al-Hassan"],
    },
    {
        "id": "Q04",
        "category": "filter",
        "difficulty": "medium",
        "query": "Find all profiles based in Paris, France.",
        "ground_truth_hints": ["Sophie Marchand", "Yuki Tanaka", "Alice Fontaine", "David Laurent"],
    },

    # ─── Category 3: Semantic / Conceptual Search ─────────────────────────────
    {
        "id": "Q05",
        "category": "semantic",
        "difficulty": "medium",
        "query": "I'm selling a developer productivity platform. Who are the best targets?",
        "ground_truth_hints": ["Thomas Keller", "Julia Novak", "Matteo Romano"],
    },
    {
        "id": "Q06",
        "category": "semantic",
        "difficulty": "medium",
        "query": "Which profiles mention pain points around data quality or data silos?",
        "ground_truth_hints": ["Raj Patel", "Astrid Müller", "Ingrid Larsson"],
    },

    # ─── Category 4: Synthesis / Cross-profile Reasoning ─────────────────────
    {
        "id": "Q07",
        "category": "synthesis",
        "difficulty": "hard",
        "query": "Compare the DISC profiles of the two CEOs/founders in the dataset and explain how you'd approach them differently in a cold email.",
        "ground_truth_hints": ["Elena Vasquez", "David Laurent", "D profile", "direct", "results", "different approach"],
    },
    {
        "id": "Q08",
        "category": "synthesis",
        "difficulty": "hard",
        "query": "Which profiles would be most receptive to a pitch about an AI-powered observability platform, and why?",
        "ground_truth_hints": ["Omar Sheikh", "Thomas Keller", "Matteo Romano", "observability", "reliability"],
    },

    # ─── Category 5: Aggregation / Stats ──────────────────────────────────────
    {
        "id": "Q09",
        "category": "aggregation",
        "difficulty": "medium",
        "query": "How many profiles are located in Ireland, and what companies do they work for?",
        "ground_truth_hints": ["3", "DHL", "HubSpot", "Stripe"],
    },
    {
        "id": "Q10",
        "category": "aggregation",
        "difficulty": "hard",
        "query": "Rank the top 3 profiles by seniority and years of experience combined, and summarize why each would be a high-value target for enterprise software sales.",
        "ground_truth_hints": ["Bernhard Koch", "Liam O'Brien", "Fatima Al-Hassan", "enterprise", "seniority"],
    },
]
