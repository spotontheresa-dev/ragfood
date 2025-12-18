#!/usr/bin/env python3
"""Quick test of the migrated RAG system"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from upstash_vector import Index
from groq import Groq

print("🧪 Testing RAG-Food Cloud Migration")
print("=" * 50)

# Test 1: Upstash Vector connection
print("\n1️⃣ Testing Upstash Vector connection...")
try:
    vector_index = Index(
        url=os.getenv("UPSTASH_VECTOR_REST_URL"),
        token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    )
    info = vector_index.info()
    print(f"   ✅ Connected to Upstash Vector")
    print(f"   Index: {info.dimension} dimensions, {info.vector_count} vectors")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 2: Ingest sample data if empty
if info.vector_count == 0:
    print("\n2️⃣ Ingesting sample data...")
    import json
    with open("foods.json", "r") as f:
        foods = json.load(f)
    
    vectors = []
    for item in foods[:10]:  # Just first 10 for quick test
        text = item["text"]
        if "region" in item:
            text += f" This food is popular in {item['region']}."
        vectors.append((item["id"], text, {"original_text": item["text"]}))
    
    vector_index.upsert(vectors=vectors)
    print(f"   ✅ Upserted {len(vectors)} vectors")
else:
    print(f"\n2️⃣ Skipping ingestion ({info.vector_count} vectors already exist)")

# Test 3: Query vector DB
print("\n3️⃣ Testing vector search...")
try:
    results = vector_index.query(
        data="What is masala dosa?",
        top_k=3,
        include_metadata=True
    )
    print(f"   ✅ Query returned {len(results)} results")
    for i, r in enumerate(results):
        text = r.metadata.get("original_text", "N/A") if r.metadata else "N/A"
        print(f"   {i+1}. ID={r.id}, Score={r.score:.4f}")
        print(f"      Text: {text[:60]}...")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 4: Groq LLM
print("\n4️⃣ Testing Groq LLM generation...")
try:
    groq_client = Groq()
    
    # Build context from results
    context = "\n".join([r.metadata.get("original_text", "") for r in results if r.metadata])
    
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful food expert. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: What is masala dosa?\n\nAnswer:"}
        ],
        max_completion_tokens=200,
        stream=False
    )
    
    answer = completion.choices[0].message.content.strip()
    print(f"   ✅ Groq response received")
    print(f"   Answer: {answer[:200]}...")
    print(f"   Tokens used: {completion.usage.total_tokens}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("✅ All tests passed! Migration successful.")
print("\nRun the full RAG system with:")
print("   python3 rag_run_cloud.py")
