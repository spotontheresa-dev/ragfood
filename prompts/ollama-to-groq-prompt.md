Create a detailed plan to migrate from local Ollama LLM to Groq Cloud API. I have added the GROQ_API_KEY to the .env file.

Current Implementation:
- Using Ollama locally with llama3.2 model
- Direct HTTP requests to localhost:11434/api/generate
- Streaming disabled for simplicity
- Local inference with no API costs

Target Implementation:
- Groq Cloud API with llama-3.1-8b-instant model
- HTTP requests to Groq's API endpoints
- API key authentication required
- Cloud-based inference with usage-based pricing

Groq API Details:
- Model: "llama-3.1-8b-instant"
- Endpoint: Groq's REST API
- Authentication: Bearer token with API key
- Response format: Similar to OpenAI API
- Rate limits and usage tracking

Code Example Reference:
```python
from groq import Groq
client = Groq()
completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": ""}],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None
)
```

Please provide:
- Detailed migration steps
- Code changes required
- Error handling for API failures
- Rate limiting considerations
- Cost implications and usage monitoring
- Fallback strategies
- Testing approach
- Performance comparison expectations

Use code example as below
from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
