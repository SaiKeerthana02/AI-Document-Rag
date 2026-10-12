# 📄 AI Document Assistant (RAG Pipeline)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://saikeerthana-doc-rag.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/SaiKeerthana02/AI-Document-Rag)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?logo=google)](https://ai.google.dev/)
[![Vector Store](https://img.shields.io/badge/Vector%20Store-FAISS-0467DF)](https://github.com/facebookresearch/faiss)

An intelligent Document Question-Answering application built using **Retrieval-Augmented Generation (RAG)**, **Google Gemini 2.5 Flash**, **FAISS Vector Indexing**, and **Streamlit**.

The application parses complex multi-page PDF/text documents (such as resumes, analytical reports, and manuals), generates high-dimensional semantic embeddings, performs fast similarity vector search, and delivers grounded, hallucination-free answers in real time.

---

## 🌐 Live Demo

Try the live application here:  
👉 **[saikeerthana-doc-rag.streamlit.app](https://saikeerthana-doc-rag.streamlit.app)**

---

## 💻 Tech Stack

| Category | Tools / Technologies |
| :--- | :--- |
| **Language** | Python 3.11 |
| **User Interface** | Streamlit |
| **Large Language Model** | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| **Text Embeddings** | Google GenAI Embeddings (`gemini-embedding-001`) |
| **Vector Database / Index** | FAISS (Facebook AI Similarity Search - `faiss-cpu`) |
| **PDF Extraction** | `pypdf` |
| **Configuration & Secrets** | `python-dotenv`, Streamlit Cloud Secrets |

---

## 🚀 Key Features

- **Document Ingestion:** Multi-page PDF text extraction powered by `pypdf`.
- **Context-Preserving Chunking:** 1200-character windows with a 200-character sliding overlap to maintain structural and semantic coherence across sections.
- **Vector Search (FAISS):** Embeddings generated via `gemini-embedding-001` and indexed locally in-memory using `faiss-cpu (IndexFlatL2)`.
- **Grounded Generation:** Low-latency, accurate answers synthesized using `gemini-2.5-flash` with strict anti-hallucination system instructions.
- **Context Transparency:** Expandable UI viewer displaying retrieved source chunks directly beneath each response.
- **Session Reset:** Interactive chat history management with an instant **Clear Chat** action.

---
```text
## 🛠️ Architecture & Workflow

[PDF / TXT Document] 
       │
       ▼
[Text Extraction (pypdf)]
       │
       ▼
[Chunking (1200 chars / 200 overlap)]
       │
       ▼
[Embedding Generation (gemini-embedding-001)]
       │
       ▼
[FAISS Vector Store (IndexFlatL2)]
       │
       ├─── [User Query] ──► [Query Embedding]
       ▼                           │
[Top-k Similarity Search] ◄────────┘
       │
       ▼
[Context + Query Prompt] ──► [Gemini 2.5 Flash] ──► [Grounded Answer + UI]
```

---

## 📂 Project Structure

```text
AI-Document-Rag/
├── app.py              # Main Streamlit app and RAG execution pipeline
├── requirements.txt    # Production dependencies for cloud deployment
├── .gitignore          # Excludes environment files and caches
├── .env                # Local secrets (API keys)
└── README.md           # Project architecture and setup documentation
```

---

## ⚙️ Local Setup & Run

1. Clone the Repository:
   git clone https://github.com/SaiKeerthana02/AI-Document-Rag.git
   cd AI-Document-Rag

2. Create and Activate a Virtual Environment:
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate

3. Install Dependencies:
   pip install -r requirements.txt

4. Set Up Environment Variables:
   Create a .env file in the root directory:
   GEMINI_API_KEY=your_gemini_api_key_here

5. Start the Application:
   streamlit run app.py