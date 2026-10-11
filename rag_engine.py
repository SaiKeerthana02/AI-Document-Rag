import os
import time
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai

# 1. Environment setup
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY dorakaledu! .env file check chey.")

client = genai.Client(api_key=api_key)

# 2. Text Chunking
def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# 3. Vector Embeddings
def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    if hasattr(response, "embedding") and response.embedding:
        return response.embedding.values
    elif hasattr(response, "embeddings") and response.embeddings:
        return response.embeddings[0].values
    else:
        raise AttributeError("Embedding values access cheyadam kudharaledu.")

# Document Data
sample_data = """
Google Gemini is a next-generation AI model developed by Google.
FAISS stands for Facebook AI Similarity Search, used for vector clustering and dense vector search.
RAG stands for Retrieval-Augmented Generation, which feeds relevant external facts to an LLM.
Python is an interpreted, high-level, general-purpose programming language.
"""

print("1. Chunking text...")
chunks = chunk_text(sample_data)

print("2. Indexing chunks into FAISS...")
embeddings = [get_embedding(c) for c in chunks]
embeddings_array = np.array(embeddings).astype("float32")

dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)
print(f"Total chunks indexed: {index.ntotal}\n")

# 4. Semantic Search (Retrieval)
query = "What is FAISS used for?"
print(f"Question: '{query}'")

query_vector = np.array([get_embedding(query)]).astype("float32")
top_k = 1
distances, indices = index.search(query_vector, top_k)

best_chunk_index = indices[0][0]
retrieved_context = chunks[best_chunk_index]
print(f"Retrieved Context Chunk:\n\"{retrieved_context}\"\n")

# 5. Generation with Gemini (with 503 retry support)
prompt = f"""
Answer the question strictly using only the provided context below.

--- Retrieved Context ---
{retrieved_context}

--- Question ---
{query}
"""

print("Asking Gemini using retrieved chunk...")
max_retries = 3
for attempt in range(max_retries):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        print("\n--- Final Answer ---")
        print(response.text)
        break
    except Exception as e:
        if "503" in str(e) and attempt < max_retries - 1:
            print("Server busy, retrying in 3 seconds...")
            time.sleep(3)
        else:
            raise e