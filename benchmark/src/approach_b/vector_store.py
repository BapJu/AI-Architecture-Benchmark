"""
Vector Store — Chroma + Gemini gemini-embedding-001

Each LinkedIn profile is split into semantic chunks:
  - identity chunk (name, headline, location, company, seniority)
  - skills & background chunk (skills, education, languages, years_experience)
  - context chunk (summary, pain_points, interests)

This gives the retriever fine-grained signal rather than embedding one big blob.
"""

import time
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from google import genai


COLLECTION_NAME = "linkedin_profiles"
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_BATCH_SIZE = 20      # stay well under the 100 req/min free-tier limit
EMBED_BATCH_PAUSE = 15.0   # seconds between batches


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Wrapper so Chroma can use Gemini gemini-embedding-001."""

    def __init__(self, client: genai.Client):
        self.client = client

    def __call__(self, input: Documents) -> Embeddings:
        # Embed in small batches to stay under free-tier rate limit.
        # Each call here may contain multiple documents (Chroma passes them all at once).
        all_embeddings: Embeddings = []
        for i in range(0, len(input), EMBED_BATCH_SIZE):
            batch = input[i : i + EMBED_BATCH_SIZE]
            result = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
            )
            all_embeddings.extend([e.values for e in result.embeddings])
            if i + EMBED_BATCH_SIZE < len(input):
                time.sleep(EMBED_BATCH_PAUSE)
        return all_embeddings


def _profile_to_chunks(profile: dict) -> list[dict]:
    """
    Decompose a profile dict into 3 focused text chunks with metadata.
    """
    pid = profile["id"]
    name = profile["full_name"]

    identity = (
        f"{name} | {profile['headline']} | {profile['location']} | "
        f"Company: {profile['company']} ({profile.get('company_size', '')}) | "
        f"Industry: {profile['industry']} | Seniority: {profile['seniority']}"
    )

    edu_str = "; ".join(
        f"{e['school']} ({e['degree']})" for e in profile.get("education", [])
    )
    skills_bg = (
        f"{name} skills: {', '.join(profile.get('skills', []))} | "
        f"Experience: {profile.get('years_experience', '?')} years | "
        f"Languages: {', '.join(profile.get('languages', []))} | "
        f"Education: {edu_str}"
    )

    pain_points_str = ", ".join(profile.get("pain_points", []))
    interests_str = ", ".join(profile.get("interests", []))
    context = (
        f"{name} — {profile.get('summary', '')} | "
        f"Pain points: {pain_points_str} | "
        f"Interests: {interests_str} | "
        f"DISC: {profile.get('disc_profile', '?')}"
    )

    return [
        {"id": f"{pid}_identity", "text": identity, "profile_id": pid, "chunk_type": "identity", "full_name": name},
        {"id": f"{pid}_skills",   "text": skills_bg, "profile_id": pid, "chunk_type": "skills",   "full_name": name},
        {"id": f"{pid}_context",  "text": context,   "profile_id": pid, "chunk_type": "context",  "full_name": name},
    ]


def build_vector_store(profiles: list[dict], client: genai.Client) -> chromadb.Collection:
    """
    Build an in-memory Chroma collection from the profile list.
    Returns the collection ready for similarity search.
    """
    chroma_client = chromadb.Client()

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embedding_fn = GeminiEmbeddingFunction(client)
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    for profile in profiles:
        all_chunks.extend(_profile_to_chunks(profile))

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[
            {
                "profile_id": c["profile_id"],
                "chunk_type": c["chunk_type"],
                "full_name": c["full_name"],
            }
            for c in all_chunks
        ],
    )

    return collection


def retrieve(collection: chromadb.Collection, query: str, top_k: int = 6) -> list[dict]:
    """
    Semantic search: returns top-k chunks most relevant to the query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "profile_id": meta["profile_id"],
            "full_name": meta["full_name"],
            "chunk_type": meta["chunk_type"],
            "similarity": round(1 - dist, 4),
        })

    return chunks
