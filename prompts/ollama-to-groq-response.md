I've created a comprehensive migration plan for moving from Ollama to Groq Cloud API at groq-migration-design.md.

Key Highlights:

Performance Improvements
- Speed: ~10x faster (2-5s → 0.3-1s response time)
- Model Size: 8B parameters (Groq) vs 1B (Ollama) = better reasoning
- Tokens/second: ~500-750 tok/s (Groq) vs ~20-50 tok/s (Ollama)

Code Changes Required
1. Replace requests.post() to Ollama with groq_client.chat.completions.create()
2. Convert simple prompt format to OpenAI-compatible chat messages
3. Add streaming support with real-time output
4. Remove local embedding dependency (handled by Upstash)

Cost Analysis
- Free tier: 500K tokens/day, 30 requests/minute
- Your estimated usage: ~75K tokens/day (easily within free tier)
- If paid: ~$0.13/month at 100 queries/day

Error Handling
- Rate limit handling with exponential backoff
- Connection retry logic (3 attempts)
- Graceful fallback to Ollama if Groq unavailable
- Response caching for repeated queries

Complete Migrated Code
The document includes a full rag_run.py rewrite with:
- Groq client initialization
- Streaming responses
- Error handling wrapper class
- Usage tracking
- Rate limiting implementation

Testing Approach
- Unit tests for API connectivity
- Integration tests for RAG pipeline
- Performance benchmarks
- Quality comparison tests