import os
import streamlit as st
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# 1. Config & API Client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="AI Document RAG", page_icon="📄", layout="wide")
st.title("📄 AI Document Assistant (RAG)")

if not api_key:
    st.error("GEMINI_API_KEY dorakaledu! .env file check chey.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Helper Functions
def extract_text(uploaded_file) -> str:
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages_text = []
        for i, page in enumerate(reader.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                pages_text.append(f"[Page {i}] {txt.strip()}")
        return "\n\n".join(pages_text)
    else:
        return uploaded_file.read().decode("utf-8")

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

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
        raise AttributeError("Failed to extract embedding values.")

def generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are an AI Document Assistant. Answer the question accurately using ONLY the provided context.
If the answer is not present in the context, say: "Document lo ee information ledu."

--- Context ---
{context}

--- Question ---
{query}
"""
    res = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )
    return res.text

# 3. Sidebar - Document Upload & Processing
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file and ("index" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name):
        with st.spinner("Processing document & building FAISS index..."):
            doc_text = extract_text(uploaded_file)
            chunks = chunk_text(doc_text)
            
            embeddings = [get_embedding(c) for c in chunks]
            embeddings_array = np.array(embeddings).astype("float32")

            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)

            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
            st.session_state["current_file"] = uploaded_file.name
            st.session_state["chat_history"] = []
            st.success(f"Indexed {len(chunks)} chunks successfully!")

# 4. Chat Interface
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input("Ask a question about your document...")

if user_query:
    if "index" not in st.session_state:
        st.warning("Please upload a document first!")
    else:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                query_vector = np.array([get_embedding(user_query)]).astype("float32")
                top_k = min(8, len(st.session_state["chunks"]))
                distances, indices = st.session_state["index"].search(query_vector, top_k)

                retrieved_chunks = [st.session_state["chunks"][i] for i in indices[0]]
                separator = "\n" + ("-" * 10) + "\n"
                combined_context = separator.join(retrieved_chunks)

                answer = generate_answer(user_query, combined_context)
                st.write(answer)

                with st.expander("View Retrieved Context Chunks"):
                    for c in retrieved_chunks:
                        st.caption(c)

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})