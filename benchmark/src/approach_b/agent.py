"""
Approach B — LangGraph ReAct Agent with Vector RAG

Architecture:
  - LLM: Gemini 2.0 Flash via langchain-google-genai
  - Tools: search_profiles (semantic search), get_profile_detail (exact lookup)
  - Pattern: ReAct (Reason → Act → Observe → Reason → ...)
  - State: tracked by LangGraph MessageGraph

The agent decides HOW MANY tool calls to make and WHICH profiles to retrieve
before answering — unlike Approach A which blindly stuffs everything.
"""

import json
import time
from typing import Annotated

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent


SYSTEM_PROMPT = """You are a B2B sales intelligence assistant backed by a LinkedIn profile database.

When answering:
1. Use search_profiles to find relevant contacts semantically.
2. Use get_profile_detail when you need complete information about a specific person.
3. Be precise — only claim facts you retrieved from the tools.
4. For filtering/ranking tasks, call search_profiles with different angles if needed.

Always ground your answer in the retrieved data."""


def build_agent_tools(collection: chromadb.Collection, profiles_by_id: dict):
    """
    Factory: creates tool functions bound to the vector store and profile lookup map.
    Returns the list of LangChain tool objects.
    """

    @tool
    def search_profiles(query: str, top_k: int = 5) -> str:
        """
        Semantic search over LinkedIn profiles. Use this to find profiles matching
        a description, skill set, industry, pain point, or any free-text criteria.

        Args:
            query: Natural language description of what you're looking for.
            top_k: Number of relevant chunks to retrieve (default 5, max 10).
        """
        from benchmark.src.approach_b.vector_store import retrieve
        top_k = min(max(int(top_k), 1), 10)
        chunks = retrieve(collection, query, top_k=top_k)
        if not chunks:
            return "No relevant profiles found."
        lines = []
        for c in chunks:
            lines.append(f"[{c['profile_id']} | {c['full_name']} | {c['chunk_type']} | sim={c['similarity']}]\n{c['text']}")
        return "\n\n".join(lines)

    @tool
    def get_profile_detail(profile_id: str) -> str:
        """
        Retrieve the complete structured profile for a given profile ID (e.g. P001).
        Use this after search_profiles identifies a candidate you need more data on.

        Args:
            profile_id: Profile ID string like 'P001', 'P015', etc.
        """
        profile = profiles_by_id.get(profile_id.upper())
        if not profile:
            return f"Profile {profile_id} not found."
        return json.dumps(profile, ensure_ascii=False, indent=2)

    return [search_profiles, get_profile_detail]


def run_agent_query(
    llm: ChatGoogleGenerativeAI,
    collection: chromadb.Collection,
    profiles_by_id: dict,
    query: str,
) -> dict:
    """
    Run the ReAct agent on a single query.
    Returns answer + metrics comparable to Approach A.
    """
    tools = build_agent_tools(collection, profiles_by_id)
    agent = create_react_agent(llm, tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    t0 = time.perf_counter()
    result = agent.invoke({"messages": messages})
    latency = time.perf_counter() - t0

    # Extract final AI message
    final_message = None
    tool_call_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage":
            final_message = msg
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                total_input_tokens += msg.usage_metadata.get("input_tokens", 0)
                total_output_tokens += msg.usage_metadata.get("output_tokens", 0)
            # Count tool calls in this message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call_count += len(msg.tool_calls)

    answer = final_message.content if final_message else "No answer generated."

    return {
        "approach": "B_agent",
        "answer": answer,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "latency_s": round(latency, 3),
        "tool_calls": tool_call_count,
    }
