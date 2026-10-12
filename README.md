# 📄 AI Document Assistant (RAG)

An intelligent Document Question-Answering system built using **Retrieval-Augmented Generation (RAG)**, **Google Gemini 2.5 Flash**, **FAISS Vector Indexing**, and **Streamlit**.

The application parses structured PDF/Text documents (such as resumes, reports, and manuals), generates semantic vector embeddings, performs top-k similarity searches, and provides factual, grounded answers using real-time LLM synthesis.

---

## 🚀 Key Features

- **Document Ingestion:** Multi-page PDF text extraction powered by `pypdf`.
- **Chunking Strategy:** 1200-character chunks with 200-character sliding overlap to maintain structural and semantic context.
- **Vector Search (FAISS):** Embeddings generated via `gemini-embedding-001` and indexed locally with `faiss-cpu (IndexFlatL2)`.
- **Grounded Generation:** Low-latency, accurate Q&A responses using `gemini-2.5-flash` with explicit hallucination-prevention prompts.
- **Context Transparency:** Expandable UI drawer displaying retrieved context chunks and relevance scores.
- **Session Management:** In-memory chat history with a dedicated **Clear Chat** reset action.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** Streamlit
- **LLM & Embeddings:** Google GenAI SDK (`gemini-2.5-flash`, `gemini-embedding-001`)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **PDF Extraction:** `pypdf`
- **Environment Management:** `python-dotenv`

---

## 📂 Project Structure

```text
AI-Document-Rag/
├── app.py              # Main Streamlit application and RAG pipeline
├── requirements.txt    # Project dependencies
├── .env                # API keys (ignored in version control)
├── .gitignore          # Git ignore specifications
└── README.md           # Project documentation