import sys, os, json, traceback
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from google import genai
from benchmark.src.approach_a.naive import run_naive_query

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

with open("benchmark/data/profiles.json") as f:
    profiles = json.load(f)

print("Testing single naive query...")
try:
    result = run_naive_query(client, profiles, "What is Sophie Marchand's current role?")
    print("Answer:", result["answer"][:200])
    print("Tokens:", result["total_tokens"])
    print("Latency:", result["latency_s"], "s")
except Exception as e:
    traceback.print_exc()
