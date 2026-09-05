"""Proves the Gemini key works end to end: one embedding call, one generation call."""

from google import genai

from src import config

client = genai.Client(api_key=config.active_api_key())

emb = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Revenue grew 20% year over year",
)
print("embedding dimensions:", len(emb.embeddings[0].values))

resp = client.models.generate_content(
    model=config.GEMINI_GENERATION_MODEL,
    contents="Reply with exactly: CONNECTION OK",
)
print("model says:", resp.text)
