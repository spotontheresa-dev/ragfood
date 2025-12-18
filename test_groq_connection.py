#!/usr/bin/env python3
"""Test Groq API connectivity"""

import os
from dotenv import load_dotenv

load_dotenv()

# Verify API key is set
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY not found in .env file")
    exit(1)

print(f"✅ GROQ_API_KEY found (starts with: {api_key[:10]}...)")

try:
    from groq import Groq
    
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say 'Hello from Groq!' in exactly those words."}],
        max_completion_tokens=50
    )
    
    answer = response.choices[0].message.content
    print(f"✅ Groq API connected successfully!")
    print(f"   Model: llama-3.1-8b-instant")
    print(f"   Response: {answer}")
    print(f"   Usage: {response.usage.total_tokens} tokens")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)
