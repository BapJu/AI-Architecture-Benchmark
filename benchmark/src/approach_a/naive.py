"""
Approach A — Naive Context Stuffing (mirrors Helloo.in production pattern)

The original Helloo.in code does:
    formatted_prompt = prompt_query + json.dumps(profile_data)

For the cold email step it concatenates ALL analyses:
    prompt = instructions + json.dumps(profile_data) + disc + jargon + soncas + json.dumps(form_data) + format_instructions

This approach sends the ENTIRE profile corpus on every query — no retrieval,
no filtering, just raw context injection. Clean, simple, but doesn't scale.
"""

import json
import time
from google import genai
from google.genai import types
from google.genai.errors import ServerError # Ensure this is imported
import random

MODEL = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """You are a B2B sales intelligence assistant.
You have access to a database of LinkedIn profiles. Answer questions about these profiles accurately and concisely.
Base your answers strictly on the provided data."""


def build_naive_prompt(profiles: list[dict], query: str) -> str:
    """
    Replicates the Helloo.in pattern: raw JSON of ALL profiles appended to the query.
    This is the 'naive context stuffing' approach from the production codebase.
    """
    profiles_json = json.dumps(profiles, ensure_ascii=False, indent=2)
    prompt = f"""Here is the complete LinkedIn profile database:

{profiles_json}

---
User query: {query}

Answer based strictly on the profiles above."""
    return prompt


def run_naive_query(
    client: genai.Client,
    profiles: list[dict],
    query: str,
    max_retries: int = 4,      # How many times to try before giving up
    base_delay: float =40.0,   # Base wait time in seconds
) -> dict:
    """
    Single-shot query with full context stuffing.
    Returns answer text + metrics (tokens, latency).
    Includes retry logic for transient 503 errors.
    """
    prompt = build_naive_prompt(profiles, query)

    t0 = time.perf_counter()
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )
            # If successful, break out of the retry loop
            break 
            
        except ServerError as e:
            print(e)
            # Check if it's specifically a 503 error
            if e.code == 503 and attempt < max_retries - 1:
                # Calculate backoff: 2s, 4s, 8s... plus a little random jitter to prevent thundering herd
                delay = (base_delay ** attempt) + random.uniform(0, 1)
                print(f"  [!] 503 Error. Retrying attempt {attempt + 2}/{max_retries} in {delay:.2f}s...")
                time.sleep(delay)
            else:
                # If it's a different ServerError or we ran out of retries, crash gracefully
                raise

    latency = time.perf_counter() - t0

    usage = response.usage_metadata
    return {
        "approach": "A_naive",
        "answer": response.text,
        "input_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
        "total_tokens": usage.total_token_count,
        "latency_s": round(latency, 3),
        "tool_calls": 0,
    }
