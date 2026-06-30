import sys, os
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Models supporting generateContent:")
for m in client.models.list():
    actions = getattr(m, "supported_actions", [])
    if "generateContent" in str(actions):
        print(f"  {m.name}")

print()
# Quick live test with the first available model
print("Testing gemini-2.0-flash...")
try:
    r = client.models.generate_content(model="gemini-2.0-flash", contents="Say OK")
    print("gemini-2.0-flash: OK —", r.text[:50])
except Exception as e:
    print("gemini-2.0-flash FAILED:", str(e)[:120])

print("Testing gemini-2.5-flash...")
try:
    r = client.models.generate_content(model="gemini-2.5-flash", contents="Say OK")
    print("gemini-2.5-flash: OK —", r.text[:50])
except Exception as e:
    print("gemini-2.5-flash FAILED:", str(e)[:120])
