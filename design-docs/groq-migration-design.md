# Ollama to Groq Cloud API Migration Design Document

**Project:** RAG-Food LLM Backend Migration  
**Date:** December 17, 2025  
**Status:** Design Phase  
**Target:** Production-ready implementation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Comparison](#architecture-comparison)
3. [Current Implementation Analysis](#current-implementation-analysis)
4. [Groq API Overview](#groq-api-overview)
5. [Detailed Migration Steps](#detailed-migration-steps)
6. [Code Changes Required](#code-changes-required)
7. [Error Handling Strategy](#error-handling-strategy)
8. [Rate Limiting Considerations](#rate-limiting-considerations)
9. [Cost Implications](#cost-implications)
10. [Fallback Strategies](#fallback-strategies)
11. [Testing Approach](#testing-approach)
12. [Performance Comparison](#performance-comparison)
13. [Implementation Checklist](#implementation-checklist)

---

## Executive Summary

This document outlines the migration from local Ollama LLM inference to Groq Cloud API for the RAG-Food application. The primary benefits are:

- **Dramatically Faster Inference:** Groq's LPU (Language Processing Unit) provides industry-leading inference speeds (~500 tokens/second)
- **No Local GPU Required:** Offload inference to cloud, freeing local resources
- **Larger Model Capacity:** Access llama-3.1-8b-instant (8B params) vs current llama3.2:1b (1B params)
- **Better Response Quality:** Larger model with more parameters yields more coherent answers
- **Simplified Deployment:** No Ollama server dependency for inference

**Key Trade-offs:**
- Network dependency for all LLM calls
- API usage costs (though Groq has generous free tier)
- Rate limits to manage
- External service dependency

---

## Architecture Comparison

### Before: Local Ollama Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG-Food Application                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┴──────────────┐
       │                          │
       ▼                          ▼
┌─────────────────┐      ┌──────────────────┐
│  Ollama Server  │      │  ChromaDB/       │
│  (localhost)    │      │  Upstash Vector  │
│                 │      │                  │
│ llama3.2:1b     │      │ Vector Storage   │
│ mxbai-embed     │      │                  │
│                 │      │                  │
│ Port: 11434     │      │                  │
└─────────────────┘      └──────────────────┘
       │
       ├── /api/embeddings (embedding generation)
       └── /api/generate  (LLM inference)

Data Flow:
1. User asks question
2. Embed question via Ollama (/api/embeddings)
3. Search vector DB for relevant context
4. Build prompt with context + question
5. Generate answer via Ollama (/api/generate)
6. Return answer to user
```

**Current Limitations:**
- llama3.2:1b is a small model (1B parameters)
- Limited reasoning capability
- Local GPU/CPU bottleneck
- Must keep Ollama server running
- ~2-5 seconds inference time on typical hardware

---

### After: Groq Cloud Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG-Food Application                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┴──────────────┐
       │                          │
       ▼                          ▼
┌─────────────────┐      ┌──────────────────┐
│  Groq Cloud API │      │  Upstash Vector  │
│  (HTTPS)        │      │  (Cloud)         │
│                 │      │                  │
│ llama-3.1-8b-   │      │ Vector Storage   │
│ instant         │      │ Auto-embedding   │
│                 │      │                  │
│ Bearer Token    │      │                  │
│ Authentication  │      │                  │
└─────────────────┘      └──────────────────┘
       │
       └── chat.completions.create (OpenAI-compatible)

Data Flow:
1. User asks question
2. Query Upstash Vector (auto-embedding)
3. Retrieve relevant context
4. Build chat messages with context + question
5. Call Groq API (chat completions)
6. Stream or return complete answer
7. Display to user
```

**Key Improvements:**
- llama-3.1-8b-instant (8B parameters) - 8x larger
- ~500 tokens/second inference speed
- No local GPU requirements
- OpenAI-compatible API format
- ~100-300ms response time

---

## Current Implementation Analysis

### Code Structure (rag_run.py)

```python
# Current LLM Configuration
LLM_MODEL = "llama3.2:1b"

# Current LLM Call (lines 82-89)
response = requests.post("http://localhost:11434/api/generate", json={
    "model": LLM_MODEL,
    "prompt": prompt,
    "stream": False
})
return response.json()["response"].strip()
```

### Current Prompt Format

```python
prompt = f"""Use the following context to answer the question.

    Context:
    {context}

    Question: {question}
    Answer:"""
```

### Dependencies
- `requests` library for HTTP calls
- Ollama server running on localhost:11434
- No authentication required (local)

---

## Groq API Overview

### Authentication
```python
# Environment variable (automatically picked up by Groq SDK)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Or explicit initialization
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
```

### API Format (OpenAI-Compatible)
```python
from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful food expert assistant."
        },
        {
            "role": "user",
            "content": "What is masala dosa?"
        }
    ],
    temperature=0.7,
    max_completion_tokens=1024,
    top_p=1,
    stream=False
)

answer = completion.choices[0].message.content
```

### Available Models

| Model | Parameters | Speed | Best For |
|-------|-----------|-------|----------|
| llama-3.3-70b-versatile | 70B | ~330 tok/s | Complex reasoning |
| llama-3.1-8b-instant | 8B | ~750 tok/s | Fast responses |
| llama-3.2-3b-preview | 3B | ~1000 tok/s | Quick tasks |
| mixtral-8x7b-32768 | 8x7B | ~480 tok/s | Long context |
| gemma2-9b-it | 9B | ~500 tok/s | Instruction following |

**Recommended:** `llama-3.1-8b-instant` for balance of speed and quality

### Rate Limits (Free Tier)

| Limit Type | Value |
|------------|-------|
| Requests per minute | 30 |
| Requests per day | 14,400 |
| Tokens per minute | 6,000 |
| Tokens per day | 500,000 |

---

## Detailed Migration Steps

### Phase 1: Environment Setup (15 minutes)

#### Step 1.1: Install Groq SDK
```bash
pip3 install groq
```

#### Step 1.2: Verify .env Configuration
```bash
# .env file should contain:
GROQ_API_KEY=gsk_your_api_key_here

# Also include (from Upstash migration):
UPSTASH_VECTOR_REST_URL=...
UPSTASH_VECTOR_REST_TOKEN=...
```

#### Step 1.3: Update requirements.txt
```
groq>=0.4.0
python-dotenv>=1.0.0
upstash-vector>=0.3.0
requests>=2.31.0
```

#### Step 1.4: Test Connectivity
```python
# test_groq_connection.py
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

try:
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say hello!"}],
        max_completion_tokens=50
    )
    print(f"✅ Groq connected: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

---

### Phase 2: Code Migration (30 minutes)

#### Step 2.1: Update Imports

**Before:**
```python
import os
import json
import chromadb
import requests
```

**After:**
```python
import os
import json
from groq import Groq
from dotenv import load_dotenv
from upstash_vector import Index
# requests no longer needed for LLM (keep if using for other purposes)
```

#### Step 2.2: Update Constants

**Before:**
```python
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "foods"
JSON_FILE = "foods.json"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2:1b"
```

**After:**
```python
load_dotenv()

JSON_FILE = "foods.json"
GROQ_MODEL = "llama-3.1-8b-instant"

# Initialize Groq client
groq_client = Groq()  # Automatically uses GROQ_API_KEY from env

# Initialize Upstash Vector (from previous migration)
vector_index = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)
```

#### Step 2.3: Create Groq LLM Function

**Before (Ollama):**
```python
# Step 6: Generate answer with Ollama
response = requests.post("http://localhost:11434/api/generate", json={
    "model": LLM_MODEL,
    "prompt": prompt,
    "stream": False
})

# Step 7: Return final result
return response.json()["response"].strip()
```

**After (Groq):**
```python
def generate_with_groq(prompt: str, context: str, question: str) -> str:
    """Generate answer using Groq Cloud API"""
    
    system_message = """You are a helpful food expert assistant. 
Answer questions based on the provided context. 
Be concise and accurate. If the context doesn't contain 
relevant information, say so."""
    
    user_message = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        raise
```

#### Step 2.4: Update RAG Query Function

**Before:**
```python
def rag_query(question):
    # Step 1: Embed the user question
    q_emb = get_embedding(question)

    # Step 2: Query the vector DB
    results = collection.query(query_embeddings=[q_emb], n_results=3)

    # Step 3: Extract documents
    top_docs = results['documents'][0]
    top_ids = results['ids'][0]
    
    # ... display logic ...
    
    # Step 5: Build prompt from context
    context = "\n".join(top_docs)
    
    prompt = f"""Use the following context to answer the question.

    Context:
    {context}

    Question: {question}
    Answer:"""

    # Step 6: Generate answer with Ollama
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    })
    
    return response.json()["response"].strip()
```

**After:**
```python
def rag_query(question: str) -> str:
    """RAG query using Upstash Vector + Groq LLM"""
    
    # Step 1: Query Upstash Vector (auto-embedding)
    results = vector_index.query(
        data=question,
        top_k=3,
        include_metadata=True
    )
    
    # Step 2: Extract documents
    top_docs = []
    top_ids = []
    for result in results:
        top_ids.append(result.id)
        top_docs.append(result.metadata.get("original_text", ""))
    
    # Step 3: Display retrieved sources
    print("\n🧠 Retrieving relevant information to reason through your question...\n")
    for i, doc in enumerate(top_docs):
        print(f"🔹 Source {i + 1} (ID: {top_ids[i]}):")
        print(f"    \"{doc}\"\n")
    print("📚 These seem to be the most relevant pieces of information to answer your question.\n")
    
    # Step 4: Build context
    context = "\n".join(top_docs)
    
    # Step 5: Generate answer with Groq
    answer = generate_with_groq(
        prompt="",  # Not used in new implementation
        context=context,
        question=question
    )
    
    return answer
```

---

### Phase 3: Streaming Support (Optional Enhancement)

#### With Streaming (Real-time output):
```python
def generate_with_groq_streaming(context: str, question: str):
    """Generate answer with streaming output"""
    
    system_message = """You are a helpful food expert assistant. 
Answer questions based on the provided context."""
    
    user_message = f"""Context:
{context}

Question: {question}

Answer:"""
    
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
        stream=True  # Enable streaming
    )
    
    print("🤖: ", end="", flush=True)
    full_response = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
        full_response += content
    print()  # Newline after streaming
    
    return full_response
```

#### Updated Interactive Loop with Streaming:
```python
# Interactive loop with streaming
print("\n🧠 RAG is ready. Ask a question (type 'exit' to quit):\n")
while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    
    # Get context from vector DB
    results = vector_index.query(data=question, top_k=3, include_metadata=True)
    context = "\n".join([r.metadata.get("original_text", "") for r in results])
    
    # Display sources
    print("\n🧠 Retrieving relevant information...\n")
    for i, r in enumerate(results):
        print(f"🔹 Source {i+1}: {r.metadata.get('original_text', '')[:50]}...")
    print()
    
    # Stream the answer
    generate_with_groq_streaming(context, question)
    print()
```

---

## Code Changes Required

### Complete Migrated rag_run.py

```python
#!/usr/bin/env python3
"""
RAG-Food: Retrieval-Augmented Generation with Upstash Vector + Groq LLM
Migrated from ChromaDB + Ollama to cloud-native stack
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from upstash_vector import Index

# Load environment variables
load_dotenv()

# Configuration
JSON_FILE = "foods.json"
GROQ_MODEL = "llama-3.1-8b-instant"

# Initialize clients
groq_client = Groq()
vector_index = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)

# Load food data
with open(JSON_FILE, "r", encoding="utf-8") as f:
    food_data = json.load(f)


def prepare_vectors(items: list) -> list:
    """Transform food items into Upstash vector format"""
    vectors = []
    for item in items:
        enriched_text = item["text"]
        if "region" in item:
            enriched_text += f" This food is popular in {item['region']}."
        if "type" in item:
            enriched_text += f" It is a type of {item['type']}."
        
        vectors.append({
            "id": item["id"],
            "data": enriched_text,
            "metadata": {
                "region": item.get("region", "Unknown"),
                "type": item.get("type", "Unknown"),
                "original_text": item["text"]
            }
        })
    return vectors


def ingest_data():
    """Ingest food data into Upstash Vector"""
    info = vector_index.info()
    if info.vector_count > 0:
        print(f"✅ {info.vector_count} documents already in Upstash Vector.")
        return
    
    print(f"🆕 Adding {len(food_data)} documents to Upstash Vector...")
    vectors = prepare_vectors(food_data)
    
    # Batch upsert
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i+BATCH_SIZE]
        vector_index.upsert(vectors=batch)
    
    print(f"✅ Successfully ingested {len(vectors)} documents.")


def generate_with_groq(context: str, question: str, stream: bool = False) -> str:
    """Generate answer using Groq Cloud API"""
    
    system_message = """You are a helpful food expert assistant. 
Answer questions based on the provided context. 
Be concise and accurate. If the context doesn't contain 
relevant information, acknowledge that."""
    
    user_message = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        if stream:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=True
            )
            
            print("🤖: ", end="", flush=True)
            full_response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                full_response += content
            print()
            return full_response
        else:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        raise


def rag_query(question: str, stream: bool = True) -> str:
    """Full RAG query pipeline"""
    
    # Step 1: Query vector database
    results = vector_index.query(
        data=question,
        top_k=3,
        include_metadata=True
    )
    
    # Step 2: Extract and display sources
    print("\n🧠 Retrieving relevant information to reason through your question...\n")
    top_docs = []
    for i, result in enumerate(results):
        doc_text = result.metadata.get("original_text", "")
        top_docs.append(doc_text)
        print(f"🔹 Source {i + 1} (ID: {result.id}):")
        print(f"    \"{doc_text}\"\n")
    
    print("📚 These seem to be the most relevant pieces of information to answer your question.\n")
    
    # Step 3: Build context
    context = "\n".join(top_docs)
    
    # Step 4: Generate answer
    if stream:
        return generate_with_groq(context, question, stream=True)
    else:
        answer = generate_with_groq(context, question, stream=False)
        return answer


def main():
    """Main entry point"""
    # Ingest data if needed
    ingest_data()
    
    # Interactive loop
    print("\n🧠 RAG is ready. Ask a question (type 'exit' to quit):\n")
    while True:
        try:
            question = input("You: ")
            if question.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            if not question.strip():
                continue
            
            answer = rag_query(question, stream=True)
            # Answer already printed if streaming
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
```

---

## Error Handling Strategy

### Comprehensive Error Handling

```python
from groq import Groq, APIError, RateLimitError, APIConnectionError
import time

class GroqLLMClient:
    """Wrapper for Groq API with error handling and retries"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.client = Groq()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model = "llama-3.1-8b-instant"
    
    def generate(self, messages: list, stream: bool = False) -> str:
        """Generate response with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_completion_tokens=1024,
                    stream=stream
                )
                
                if stream:
                    return self._handle_stream(completion)
                else:
                    return completion.choices[0].message.content.strip()
            
            except RateLimitError as e:
                wait_time = self._get_retry_after(e) or (self.retry_delay * (2 ** attempt))
                print(f"⏱️  Rate limited. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                
            except APIConnectionError as e:
                if attempt < self.max_retries - 1:
                    print(f"🔌 Connection error. Retrying ({attempt + 1}/{self.max_retries})...")
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Failed to connect to Groq after {self.max_retries} attempts: {e}")
            
            except APIError as e:
                # Non-retryable errors
                if e.status_code == 401:
                    raise ValueError("Invalid API key. Check GROQ_API_KEY in .env")
                elif e.status_code == 400:
                    raise ValueError(f"Bad request: {e.message}")
                else:
                    raise
            
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                raise
        
        raise RuntimeError("Max retries exceeded")
    
    def _handle_stream(self, completion) -> str:
        """Handle streaming response"""
        full_response = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            full_response += content
        print()
        return full_response
    
    def _get_retry_after(self, error) -> float:
        """Extract retry-after time from rate limit error"""
        try:
            # Groq returns retry-after in headers
            return float(error.response.headers.get("retry-after", 1.0))
        except:
            return None
```

### Error Types and Handling

| Error Type | Cause | Strategy |
|------------|-------|----------|
| `RateLimitError` | Too many requests | Exponential backoff with retry-after header |
| `APIConnectionError` | Network issues | Retry up to 3 times |
| `APIError (401)` | Invalid API key | Fail immediately, prompt to check key |
| `APIError (400)` | Bad request format | Fail immediately, log for debugging |
| `APIError (500)` | Server error | Retry with backoff |
| `Timeout` | Slow response | Increase timeout or retry |

---

## Rate Limiting Considerations

### Groq Rate Limits (Free Tier)

```python
# Rate limit configuration
GROQ_LIMITS = {
    "requests_per_minute": 30,
    "requests_per_day": 14_400,
    "tokens_per_minute": 6_000,
    "tokens_per_day": 500_000
}
```

### Rate Limiter Implementation

```python
import time
from collections import deque
from threading import Lock

class RateLimiter:
    """Token bucket rate limiter for Groq API"""
    
    def __init__(self, requests_per_minute: int = 30, tokens_per_minute: int = 6000):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.request_times = deque()
        self.token_usage = deque()
        self.lock = Lock()
    
    def wait_if_needed(self, estimated_tokens: int = 100):
        """Block until rate limit allows request"""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Clean old entries
            while self.request_times and self.request_times[0] < minute_ago:
                self.request_times.popleft()
            while self.token_usage and self.token_usage[0][0] < minute_ago:
                self.token_usage.popleft()
            
            # Check request limit
            if len(self.request_times) >= self.rpm:
                wait_time = self.request_times[0] - minute_ago
                print(f"⏱️  Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 0.1)
            
            # Check token limit
            current_tokens = sum(t[1] for t in self.token_usage)
            if current_tokens + estimated_tokens > self.tpm:
                wait_time = self.token_usage[0][0] - minute_ago
                print(f"⏱️  Token limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 0.1)
            
            # Record this request
            self.request_times.append(time.time())
    
    def record_usage(self, tokens: int):
        """Record token usage after completion"""
        with self.lock:
            self.token_usage.append((time.time(), tokens))

# Global rate limiter
rate_limiter = RateLimiter()

def generate_with_rate_limit(context: str, question: str) -> str:
    """Generate with rate limiting"""
    # Estimate tokens (rough: 1 token ≈ 4 characters)
    estimated_tokens = (len(context) + len(question)) // 4 + 200
    
    rate_limiter.wait_if_needed(estimated_tokens)
    
    # ... make Groq API call ...
    
    # Record actual usage
    rate_limiter.record_usage(actual_tokens)
```

### Best Practices for Rate Limits

1. **Batch Similar Queries:** Group related questions
2. **Cache Responses:** Store common Q&A pairs
3. **Use Streaming:** Appears faster, same rate limit impact
4. **Monitor Usage:** Track daily/monthly consumption
5. **Implement Backoff:** Exponential delay on 429 errors

---

## Cost Implications

### Pricing Comparison

| Aspect | Ollama (Local) | Groq (Cloud) |
|--------|---------------|--------------|
| **Infrastructure** | Your hardware | Groq's LPUs |
| **Base Cost** | $0 (electricity only) | Free tier available |
| **Scaling** | Buy more GPUs | Pay per token |
| **Maintenance** | Self-managed | Managed service |

### Groq Pricing (as of 2025)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| llama-3.1-8b-instant | $0.05 | $0.08 |
| llama-3.3-70b-versatile | $0.59 | $0.79 |
| mixtral-8x7b-32768 | $0.24 | $0.24 |

### Cost Estimation for RAG-Food

**Assumptions:**
- 100 queries per day
- Average context: 500 tokens (retrieved documents)
- Average question: 50 tokens
- Average response: 200 tokens

**Daily Cost:**
```
Input tokens:  100 queries × (500 + 50) tokens = 55,000 tokens
Output tokens: 100 queries × 200 tokens = 20,000 tokens

Daily cost: (55,000 × $0.05 + 20,000 × $0.08) / 1,000,000
          = $0.00275 + $0.0016
          = $0.00435/day
          ≈ $0.13/month
```

**Free Tier Coverage:**
- Free tier: 500K tokens/day
- Our usage: ~75K tokens/day
- **Easily within free tier** ✓

### Usage Monitoring

```python
class UsageTracker:
    """Track API usage and costs"""
    
    def __init__(self):
        self.daily_input_tokens = 0
        self.daily_output_tokens = 0
        self.total_requests = 0
        self.start_date = None
    
    def record(self, input_tokens: int, output_tokens: int):
        """Record a completed request"""
        self.daily_input_tokens += input_tokens
        self.daily_output_tokens += output_tokens
        self.total_requests += 1
    
    def get_cost_estimate(self) -> float:
        """Calculate estimated cost"""
        input_cost = self.daily_input_tokens * 0.05 / 1_000_000
        output_cost = self.daily_output_tokens * 0.08 / 1_000_000
        return input_cost + output_cost
    
    def print_summary(self):
        """Print usage summary"""
        cost = self.get_cost_estimate()
        print(f"📊 Usage Summary:")
        print(f"   Requests: {self.total_requests}")
        print(f"   Input tokens: {self.daily_input_tokens:,}")
        print(f"   Output tokens: {self.daily_output_tokens:,}")
        print(f"   Estimated cost: ${cost:.4f}")
```

---

## Fallback Strategies

### Strategy 1: Graceful Degradation to Ollama

```python
class HybridLLMClient:
    """Use Groq with Ollama fallback"""
    
    def __init__(self):
        self.groq_client = Groq()
        self.ollama_url = "http://localhost:11434/api/generate"
        self.use_groq = True
    
    def generate(self, context: str, question: str) -> str:
        """Generate with automatic fallback"""
        if self.use_groq:
            try:
                return self._generate_groq(context, question)
            except Exception as e:
                print(f"⚠️  Groq unavailable ({e}), falling back to Ollama...")
                self.use_groq = False
        
        return self._generate_ollama(context, question)
    
    def _generate_groq(self, context: str, question: str) -> str:
        """Generate using Groq"""
        completion = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a food expert."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            max_completion_tokens=1024
        )
        return completion.choices[0].message.content
    
    def _generate_ollama(self, context: str, question: str) -> str:
        """Fallback to local Ollama"""
        import requests
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        response = requests.post(self.ollama_url, json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        return response.json()["response"]
```

### Strategy 2: Response Caching

```python
import hashlib
import json
from pathlib import Path

class ResponseCache:
    """Cache LLM responses to reduce API calls"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_key(self, context: str, question: str) -> str:
        """Generate cache key from inputs"""
        content = f"{context}||{question}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, context: str, question: str) -> str | None:
        """Get cached response if exists"""
        key = self._get_key(context, question)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            print("💾 Using cached response")
            return data["response"]
        return None
    
    def set(self, context: str, question: str, response: str):
        """Cache a response"""
        key = self._get_key(context, question)
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps({
            "context": context[:100],
            "question": question,
            "response": response
        }))
```

### Strategy 3: Queue and Retry

```python
from queue import Queue
from threading import Thread
import time

class QueryQueue:
    """Queue queries during outages"""
    
    def __init__(self, llm_client):
        self.queue = Queue()
        self.llm_client = llm_client
        self.is_processing = True
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()
    
    def enqueue(self, context: str, question: str, callback):
        """Add query to queue"""
        self.queue.put((context, question, callback))
    
    def _process_queue(self):
        """Process queued queries"""
        while self.is_processing:
            try:
                context, question, callback = self.queue.get(timeout=1)
                response = self.llm_client.generate(context, question)
                callback(response)
            except Exception:
                pass
```

---

## Testing Approach

### Unit Tests

```python
# tests/test_groq_client.py
import pytest
from unittest.mock import Mock, patch
from groq import Groq

class TestGroqIntegration:
    """Test Groq API integration"""
    
    def test_connection(self):
        """Test basic connectivity"""
        client = Groq()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_completion_tokens=10
        )
        assert "test" in response.choices[0].message.content.lower()
    
    def test_rag_response_quality(self):
        """Test RAG response with food context"""
        context = "Masala dosa is a thin crispy crepe filled with spicy mashed potatoes."
        question = "What is masala dosa?"
        
        # ... call generate function ...
        
        assert "dosa" in response.lower()
        assert "potato" in response.lower() or "crispy" in response.lower()
    
    @patch('groq.Groq')
    def test_error_handling(self, mock_groq):
        """Test error handling"""
        mock_groq.return_value.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            # ... call generate function ...
            pass
```

### Integration Tests

```python
# tests/test_rag_pipeline.py

def test_full_rag_pipeline():
    """Test complete RAG flow"""
    question = "What Indian food uses chickpeas?"
    
    # Should retrieve relevant documents and generate answer
    answer = rag_query(question, stream=False)
    
    # Verify answer mentions chickpeas-related food
    keywords = ["chana", "chickpea", "chole", "chhole"]
    assert any(kw in answer.lower() for kw in keywords)

def test_streaming_output():
    """Test streaming response"""
    question = "What is biryani?"
    
    # Capture streaming output
    import io
    import sys
    
    captured = io.StringIO()
    sys.stdout = captured
    
    answer = rag_query(question, stream=True)
    
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    
    assert len(output) > 0
    assert "rice" in answer.lower() or "biryani" in answer.lower()
```

### Performance Tests

```python
# tests/test_performance.py
import time

def test_response_time():
    """Test response latency"""
    question = "What is samosa?"
    
    start = time.time()
    answer = rag_query(question, stream=False)
    elapsed = time.time() - start
    
    # Should be under 2 seconds
    assert elapsed < 2.0, f"Response took {elapsed:.2f}s"
    print(f"✓ Response time: {elapsed:.3f}s")

def test_throughput():
    """Test multiple queries"""
    questions = [
        "What is dosa?",
        "What is biryani?",
        "What fruits are yellow?",
        "What is paneer butter masala?",
        "What is samosa?"
    ]
    
    start = time.time()
    for q in questions:
        rag_query(q, stream=False)
    elapsed = time.time() - start
    
    qps = len(questions) / elapsed
    print(f"✓ Throughput: {qps:.2f} queries/second")
    assert qps > 0.5  # At least 0.5 q/s
```

### Test Runner Script

```bash
#!/bin/bash
# run_tests.sh

echo "🧪 Running Groq Migration Tests..."

# Unit tests
python -m pytest tests/test_groq_client.py -v

# Integration tests
python -m pytest tests/test_rag_pipeline.py -v

# Performance tests
python -m pytest tests/test_performance.py -v

echo "✅ All tests complete!"
```

---

## Performance Comparison

### Benchmark Results (Expected)

| Metric | Ollama (llama3.2:1b) | Groq (llama-3.1-8b-instant) |
|--------|---------------------|----------------------------|
| **Model Size** | 1B parameters | 8B parameters |
| **Time to First Token** | 500-1000ms | 50-100ms |
| **Tokens per Second** | 20-50 tok/s | 500-750 tok/s |
| **Total Response Time** | 2-5 seconds | 0.3-1 second |
| **Answer Quality** | Basic | High |
| **Context Understanding** | Limited | Good |
| **Hardware Required** | GPU/CPU locally | None (cloud) |

### Benchmark Script

```python
# benchmark.py
import time
import statistics

def benchmark_groq(questions: list, iterations: int = 3):
    """Benchmark Groq performance"""
    
    times = []
    
    for q in questions:
        for _ in range(iterations):
            start = time.time()
            answer = rag_query(q, stream=False)
            elapsed = time.time() - start
            times.append(elapsed)
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0
    }

if __name__ == "__main__":
    test_questions = [
        "What is masala dosa?",
        "Which Indian dish is made with rice?",
        "What fruit is yellow and sweet?",
        "What is paneer butter masala?",
        "What snack has potato filling?"
    ]
    
    results = benchmark_groq(test_questions)
    
    print("📊 Benchmark Results (Groq llama-3.1-8b-instant):")
    print(f"   Mean response time:   {results['mean']*1000:.1f}ms")
    print(f"   Median response time: {results['median']*1000:.1f}ms")
    print(f"   Min response time:    {results['min']*1000:.1f}ms")
    print(f"   Max response time:    {results['max']*1000:.1f}ms")
    print(f"   Std deviation:        {results['stdev']*1000:.1f}ms")
```

### Expected Improvements

1. **Speed:** ~10x faster inference (2-5s → 0.3-1s)
2. **Quality:** Better reasoning from 8B vs 1B model
3. **Consistency:** Cloud infrastructure more stable than local
4. **Scalability:** No local GPU bottleneck

### Quality Comparison

| Question | Ollama (1B) Answer | Groq (8B) Answer |
|----------|-------------------|------------------|
| "What is masala dosa?" | "Masala dosa is a food." | "Masala dosa is a thin, crispy fermented crepe from South India, traditionally filled with a spicy potato masala. It's a popular breakfast dish served with coconut chutney and sambar." |
| "Which food uses chickpeas?" | "Chana uses chickpeas." | "Several Indian dishes use chickpeas (chana). Chana masala is a popular North Indian curry made with chickpeas in a spiced tomato-based sauce. Chole bhature features spiced chickpeas served with fried bread." |

---

## Implementation Checklist

### Pre-Migration
- [ ] Verify GROQ_API_KEY is in .env file
- [ ] Test Groq API connectivity
- [ ] Backup current rag_run.py
- [ ] Install groq SDK (`pip3 install groq`)
- [ ] Update requirements.txt

### Code Changes
- [ ] Update imports (add Groq, remove Ollama-specific)
- [ ] Update constants (GROQ_MODEL)
- [ ] Initialize Groq client
- [ ] Create generate_with_groq() function
- [ ] Update rag_query() to use Groq
- [ ] Add streaming support (optional)
- [ ] Implement error handling
- [ ] Add rate limiting
- [ ] Add usage tracking

### Testing
- [ ] Test basic connectivity
- [ ] Test RAG pipeline end-to-end
- [ ] Test error handling
- [ ] Test rate limiting
- [ ] Benchmark performance
- [ ] Compare answer quality

### Deployment
- [ ] Switch main script to Groq
- [ ] Monitor first 24 hours
- [ ] Track API usage/costs
- [ ] Document any issues
- [ ] Update README.md

### Post-Deployment
- [ ] Archive Ollama configuration (keep as fallback)
- [ ] Optimize rate limiting
- [ ] Implement caching if needed
- [ ] Set up usage alerts

---

## Conclusion

Migrating from Ollama to Groq Cloud API provides significant benefits:

1. **~10x Faster Response:** 0.3-1s vs 2-5s
2. **8x Larger Model:** Better reasoning and accuracy
3. **No Local Infrastructure:** Reduced complexity
4. **Free Tier Coverage:** ~$0.13/month at moderate usage
5. **Easy Fallback:** Can revert to Ollama if needed

The migration is straightforward with the main changes being:
- Replace `requests.post()` to Ollama with `groq_client.chat.completions.create()`
- Convert from simple prompt to chat messages format
- Add API key authentication via environment variable
- Implement error handling for network/rate limit issues

**Recommended Timeline:** 1-2 hours for basic migration, additional 1-2 hours for error handling and testing.
