Create a vibrant, modern Food RAG (Retrieval-Augmented Generation) web application with the following specifications:

**REFERENCE PYTHON CODE:**
#!/usr/bin/env python3
"""
RAG-Food: Retrieval-Augmented Generation with Upstash Vector + Groq LLM
Migrated from ChromaDB + Ollama to cloud-native stack

Architecture:
- Vector DB: Upstash Vector (cloud-hosted, auto-embedding)
- LLM: Groq Cloud API (llama-3.1-8b-instant)
- Embedding: mixedbread-ai/mxbai-embed-large-v1 (built into Upstash)
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

# Initialize Groq client
groq_client = Groq()

# Initialize Upstash Vector
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
        
        vectors.append((
            item["id"],  # id
            enriched_text,  # data (text to embed)
            {
                "region": item.get("region", "Unknown"),
                "type": item.get("type", "Unknown"),
                "original_text": item["text"]
            }  # metadata
        ))
    return vectors


def ingest_data():
    """Ingest food data into Upstash Vector"""
    try:
        info = vector_index.info()
        if info.vector_count > 0:
            print(f"✅ {info.vector_count} documents already in Upstash Vector.")
            return
    except Exception as e:
        print(f"⚠️  Could not get index info: {e}")
    
    print(f"🆕 Adding {len(food_data)} documents to Upstash Vector...")
    vectors = prepare_vectors(food_data)
    
    # Batch upsert
    BATCH_SIZE = 100
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i+BATCH_SIZE]
        vector_index.upsert(vectors=batch)
        print(f"   Upserted batch {i//BATCH_SIZE + 1}/{(len(vectors)-1)//BATCH_SIZE + 1}")
    
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
    """Full RAG query pipeline using Upstash Vector + Groq LLM"""
    
    # Step 1: Query vector database (Upstash handles embedding automatically)
    results = vector_index.query(
        data=question,
        top_k=3,
        include_metadata=True
    )
    
    # Step 2: Extract and display sources
    print("\n🧠 Retrieving relevant information to reason through your question...\n")
    top_docs = []
    for i, result in enumerate(results):
        doc_text = result.metadata.get("original_text", "") if result.metadata else ""
        top_docs.append(doc_text)
        print(f"🔹 Source {i + 1} (ID: {result.id}):")
        print(f"    \"{doc_text}\"\n")
    
    print("📚 These seem to be the most relevant pieces of information to answer your question.\n")
    
    # Step 3: Build context
    context = "\n".join(top_docs)
    
    # Step 4: Generate answer with Groq
    if stream:
        return generate_with_groq(context, question, stream=True)
    else:
        answer = generate_with_groq(context, question, stream=False)
        print(f"🤖: {answer}")
        return answer


def main():
    """Main entry point"""
    # Verify configuration
    if not os.getenv("UPSTASH_VECTOR_REST_URL"):
        print("❌ UPSTASH_VECTOR_REST_URL not set in .env")
        return
    if not os.getenv("UPSTASH_VECTOR_REST_TOKEN"):
        print("❌ UPSTASH_VECTOR_REST_TOKEN not set in .env")
        return
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set in .env")
        return
    
    print("🚀 RAG-Food with Upstash Vector + Groq LLM")
    print("=" * 50)
    
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
            
            rag_query(question, stream=True)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()


**Technology Requirements:**
- Next.js 15.5.0 or higher (with React 19)
- Use Server Actions exclusively (no API routes)
- Shadcn UI for components
- Tailwind CSS for styling
- use groq-sdk for Groq integration
- TypeScript throughout

**Functionality Requirements:**
- Replicate the exact RAG workflow from the Python code above
- User can enter food-related questions in a chat-like interface
- Display both vector search results AND LLM-generated responses (like the Python version)
- Show loading states during processing
- Handle errors gracefully with user-friendly messages
- Responsive design that works on mobile and desktop

**Data Context:**
The application connects to an existing Upstash Vector Database containing food data like:
- "A banana is a yellow fruit that is soft and sweet." (Tropical, Fruit)
- "A lemon is yellow and very sour." (Tropical, Fruit)  
- "A chili pepper is red and extremely spicy." (Various, Spice)
- "An apple can be red, green, or yellow and has a sweet taste." (Global, Fruit)

**Environment Variables:**
- UPSTASH_VECTOR_REST_URL
- UPSTASH_VECTOR_REST_TOKEN  
- GROQ_API_KEY

**UI Requirements:**
- Attractive header with food/cooking theme
- Chat interface similar to the Python CLI experience
- Show "Sources" section with retrieved documents (like the Python print statements)
- Show "AI Response" section with generated answer
- Footer pushed to bottom of page
- Loading animations and error states
- Send button (can be icon) and Enter key support 

**Technical Details:**
- Use Upstash Vector's built-in embeddings (mxbai-embed-large-v1)
- Use Groq's llama-3.1-8b-instant model for fast responses (same as Python)
- Implement proper error handling for API failures
- Add rate limiting considerations
- Include all files needed for Vercel deployment
- No hardcoded data or mock responses

**Design:**
- Modern, clean interface with food-themed colors
- Vibrant but professional appearance
- Good contrast and accessibility
- Smooth animations and transitions
- Mobile-first responsive design

Please create a complete, deployable application that replicates the Python RAG functionality as a modern web application.

# Example Python code structure to reference:
import os
import json
from upstash_vector import Index
from dotenv import load_dotenv
from groq import Groq

# Initialize clients
upstash_index = Index.from_env()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rag_query(question, model="llama-3.1-8b-instant"):
    # Vector search
    results = upstash_index.query(
        data=question,
        top_k=3,
        include_metadata=True
    )
    
    # LLM generation  
    completion = groq_client.chat.completions.create(
        model=model,
        messages=[...],
        temperature=0.7
    )
    
    return completion.choices[0].message.content.strip()


    [
  {
    "id": "1",
    "text": "A banana is a yellow fruit that is soft and sweet.",
    "region": "Tropical",
    "type": "Fruit"
  },
  {
    "id": "2", 
    "text": "A lemon is yellow and very sour.",
    "region": "Tropical",
    "type": "Fruit"
  },
  {
    "id": "3",
    "text": "A chili pepper is red and extremely spicy.", 
    "region": "Various",
    "type": "Spice"
  },
  {
    "id": "4",
    "text": "An apple can be red, green, or yellow and has a sweet taste.",
    "region": "Global", 
    "type": "Fruit"
  }
]
This shows the structure of your food data. The AI will understand this format and create appropriate queries.