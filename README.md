# Multi-Document Research Assistant (RAG) 📚🤖

A simple, clean, and complete **Retrieval-Augmented Generation (RAG)** application built in Python that enables users to upload multiple PDF documents and ask questions about their content with grounded answers and page-level citations.

---

## 1. Project Overview

The **Multi-Document Research Assistant** lets users upload multiple PDF files (e.g., textbook chapters, research papers, resumes, reports) and query them through a natural language chat interface. Rather than relying on a general-purpose Large Language Model's parametric memory (which can hallucinate), this system extracts text from the uploaded PDFs, indexes it in a local vector database, retrieves relevant passages, and forces the LLM to generate answers strictly grounded in the retrieved facts.

---

## 2. What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI architecture pattern designed to combine the strengths of information retrieval systems with generative LLMs:

- **Retrieval**: Search an external knowledge base (e.g., custom PDFs, corporate docs) to locate relevant information for a user query.
- **Augmented Generation**: Supply those retrieved snippets as context to an LLM so it generates accurate, verifiable answers grounded in facts.

RAG eliminates model hallucinations, keeps responses up to date, and provides transparent source citations without needing costly model fine-tuning.

---

## 3. How the RAG Pipeline Works

```
📄 Upload PDFs ──> PyMuPDF ──> Recursive Chunking ──> SentenceTransformers ──> ChromaDB
                                                                                  │
User Question ──> Similarity Search (top_k) ──> Context Snippets ─────────────────┘
                                                       │
                                                       ▼
                                   Groq LLM (Llama 3.3) + Grounding Prompt
                                                       │
                                                       ▼
                                        Answer + Source & Page Citations
```

1. **Extraction**: `PyMuPDF` reads each page of uploaded PDFs, attaching metadata (filename and 1-based page number).
2. **Chunking**: `RecursiveCharacterTextSplitter` breaks text into ~1000 character snippets with 200 character overlap to maintain semantic continuity.
3. **Embeddings**: `SentenceTransformers` (`all-MiniLM-L6-v2`) converts each text chunk into a 384-dimensional dense floating-point vector representation.
4. **Vector Storage**: `ChromaDB` stores the vectors, text content, and metadata locally (`data/chroma`). Duplicate chunk hashing prevents redundant indexing.
5. **Retrieval**: When the user asks a question, the question is embedded and compared against stored chunk vectors using cosine distance similarity.
6. **LLM Generation**: The top-K retrieved chunks are sent to `Groq API` (`Llama 3.3` / `groq/compound`) or `OpenAI` alongside a system prompt requiring answers to be strictly grounded in the retrieved context. If insufficient evidence exists, it returns a standard fallback response.
7. **Citations**: Page and document citations are formatted and displayed directly below the generated answer.

---

## 4. Technology Stack

- **Python 3.11+**
- **Streamlit**: Web frontend and interactive chat interface.
- **PyMuPDF (`pymupdf`)**: Fast PDF text and page extraction.
- **LangChain (`langchain-text-splitters`)**: Recursive text chunking.
- **Sentence-Transformers (`all-MiniLM-L6-v2`)**: Local semantic text embeddings.
- **ChromaDB**: Local persistent vector database.
- **Groq API (`groq`)**: High-speed free LLM text generation (`Llama 3.3` / `groq/compound`).
- **OpenAI API (`openai`)**: Optional fallback LLM.
- **python-dotenv**: Environment variable management.
- **pytest**: Automated testing framework.

---

## 5. Project Structure

```
multi-document-rag/
│
├── app.py                  # Streamlit UI & Session State controller
├── requirements.txt        # Python package dependencies
├── .env                    # Environment variables (Groq / OpenAI API keys)
├── .env.example            # Environment template file
├── .gitignore              # Files excluded from git
├── README.md               # Documentation & setup guide
│
├── data/
│   └── chroma/             # Local persistent ChromaDB database folder
│
├── src/
│   ├── __init__.py         # Package initialization
│   ├── pdf_processor.py    # PyMuPDF PDF text & page extraction
│   ├── chunker.py          # Recursive text chunking with metadata preservation
│   ├── embeddings.py       # SentenceTransformers embedding generation
│   ├── vector_store.py     # ChromaDB vector collection management & deduplication
│   ├── retriever.py        # Similarity search retriever
│   └── generator.py        # Grounded Groq / OpenAI answer generation & citations
│
└── tests/
    └── test_basic.py       # Pytest unit & pipeline verification tests
```

---

## 6. Installation

1. **Clone or Navigate to the Project Directory**:
   ```bash
   git clone https://github.com/Ashutoshe4678/Multi-Document-Research-Assistant-RAG-.git
   cd Multi-Document-Research-Assistant-RAG-
   ```

2. **Create and Activate a Virtual Environment** *(recommended)*:
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 7. Environment Variable Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and add your **Groq API Key** (100% FREE from [console.groq.com/keys](https://console.groq.com/keys)):
   ```ini
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```

*(Optional: You can also set `OPENAI_API_KEY=sk-...` if using OpenAI).*

---

## 8. How to Run the Application

1. **Start the Streamlit Server**:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser at `http://localhost:8501`.
3. Upload one or more PDF files via the sidebar.
4. Click **⚡ Process Documents**.
5. Type your question in the chat input and receive grounded answers with citations!

---

## 9. Run Automated Tests

To run the pipeline test suite:
```bash
pytest tests/test_basic.py -v
```

---

## 10. Example Questions

Given sample AI/ML documents (`machine_learning.pdf`, `deep_learning.pdf`, `nlp.pdf`):

- *"What are the main differences between supervised and unsupervised learning?"*
- *"What is a neural network and how is it used in deep learning?"*
- *"What techniques are used for Natural Language Processing?"*

---

## 11. Limitations

- **Text-Only PDFs**: Current PyMuPDF extraction relies on selectable text layers. Image-only (scanned) PDFs without OCR are skipped.
- **Simple Chunking**: Uses recursive character splitting rather than semantic or structural section headers.
- **Basic Vector Search**: Uses dense similarity search without reranking or BM25 keyword hybrid search.

---

## 12. Future Improvements

- Integrate Tesseract OCR for scanned PDF support.
- Implement Hybrid Search (BM25 + Dense Embeddings).
- Add cross-encoder reranking (e.g. Cohere / BGE reranker) for improved retrieval accuracy.
- Support additional file formats (DOCX, Markdown, HTML).
