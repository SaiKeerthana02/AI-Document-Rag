import os
import time
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# 1. API Client setup
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY dorakaledu! .env file check chey.")

client = genai.Client(api_key=api_key)

# 2. Multi-Format Document Reader
def extract_text_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dorakaledu: {file_path}")

    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        extracted = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                extracted.append(f"[Page {page_num}] {text.strip()}")
        return "\n\n".join(extracted)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# 3. Sliding-window chunking
def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# 4. Embeddings generator
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

# 5. Gemini generation with retry logic
def generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are an AI Document Assistant. Answer the question accurately using ONLY the provided context.
If the answer is not contained in the context, say: "Document lo ee information ledu."

--- Context ---
{context}

--- Question ---
{query}
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )
            return res.text
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise e

# --- Main Pipeline Execution ---
if __name__ == "__main__":
    target_file = "sample.txt"  # Or your .pdf file path
    print(f"Loading document: {target_file}...")
    doc_text = extract_text_from_file(target_file)

    print("Chunking document...")
    chunks = chunk_text(doc_text)
    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings and building FAISS index...")
    embeddings = [get_embedding(c) for c in chunks]
    embeddings_array = np.array(embeddings).astype("float32")

    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    print("FAISS Index ready!\n")

    print("=" * 50)
    print("AI Document Assistant - Interactive Console")
    print("Type your question below (Type 'exit' or 'quit' to stop):")
    print("=" * 50)

    while True:
        try:
            user_query = input("\nYou: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Search top 2 most relevant chunks
            query_vector = np.array([get_embedding(user_query)]).astype("float32")
            top_k = min(2, len(chunks))
            distances, indices = index.search(query_vector, top_k)

            retrieved_chunks = [chunks[idx] for idx in indices[0]]
            combined_context = "\n---\n".join(retrieved_chunks)

            # Generate response
            answer = generate_answer(user_query, combined_context)
            print(f"\nAI: {answer}")

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break