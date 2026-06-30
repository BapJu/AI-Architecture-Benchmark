import sys, os, json
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from google import genai
from benchmark.src.approach_b.vector_store import build_vector_store, retrieve

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
with open("benchmark/data/profiles.json") as f:
    profiles = json.load(f)

print("Building vector store...")
col = build_vector_store(profiles, client)
print(f"Store ready ({len(profiles) * 3} chunks). Testing retrieval...")
chunks = retrieve(col, "VP Sales enterprise software Paris", top_k=3)
for c in chunks:
    print(f"  [{c['profile_id']}] {c['full_name']} | sim={c['similarity']}")
print("Vector store OK")
