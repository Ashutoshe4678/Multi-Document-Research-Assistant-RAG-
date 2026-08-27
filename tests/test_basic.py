import os
import shutil
import pytest
import pymupdf  # Modern PyMuPDF API

from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.generator import RAGGenerator, FALLBACK_RESPONSE

TEST_DB_PATH = "data/chroma_test"


@pytest.fixture(autouse=True)
def cleanup_test_db():
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
    yield
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


def create_sample_pdf_bytes(text_content: str) -> bytes:
    """Helper to generate in-memory sample PDF bytes using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text_content)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_pdf_extraction():
    """Verify PyMuPDF text extraction and page metadata indexing."""
    sample_text = "Supervised learning uses labeled datasets to train machine learning algorithms."
    pdf_bytes = create_sample_pdf_bytes(sample_text)

    pages = extract_text_from_pdf(pdf_bytes, "machine_learning.pdf")

    assert len(pages) == 1
    assert pages[0]["source"] == "machine_learning.pdf"
    assert pages[0]["page"] == 1
    assert "Supervised learning" in pages[0]["text"]


def test_chunking_metadata_retention():
    """Verify text splitting preserves source document and page number metadata."""
    documents = [{
        "text": "Supervised learning trains models on labeled data. Unsupervised learning finds patterns in unlabeled data. " * 5,
        "source": "ml_guide.pdf",
        "page": 3
    }]

    chunks = create_chunks(documents, chunk_size=150, chunk_overlap=30)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "ml_guide.pdf"
        assert chunk["metadata"]["page"] == 3
        assert "chunk_index" in chunk["metadata"]


def test_embeddings_generation():
    """Verify SentenceTransformers embedding vector output dimensions."""
    em = EmbeddingManager()
    vectors = em.embed_texts(["Hello world", "Artificial intelligence RAG system"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384  # all-MiniLM-L6-v2 vector dimension


def test_vector_store_end_to_end():
    """Verify ChromaDB document indexing, deduplication, and similarity retrieval."""
    em = EmbeddingManager()
    vs = VectorStoreManager(persist_directory=TEST_DB_PATH, collection_name="test_coll", embedding_manager=em)
    vs.clear_store()

    documents = [
        {"text": "Deep learning relies heavily on multi-layer neural networks.", "source": "deep_learning.pdf", "page": 1},
        {"text": "Natural Language Processing (NLP) enables text analysis and summarization.", "source": "nlp.pdf", "page": 2}
    ]
    chunks = create_chunks(documents)
    added_count = vs.add_chunks(chunks)
    assert added_count == len(chunks)

    # Test deduplication
    readded_count = vs.add_chunks(chunks)
    assert readded_count == 0

    # Test retriever lookup
    retriever = RAGRetriever(vs)
    results = retriever.retrieve("neural networks", top_k=2)
    assert len(results) > 0
    assert results[0]["source"] == "deep_learning.pdf"
    assert results[0]["page"] == 1


def test_generator_fallback():
    """Verify strictly grounded fallback response when context is empty."""
    generator = RAGGenerator()
    answer, citations = generator.generate_answer(query="What is quantum computing?", context_chunks=[])
    assert answer == FALLBACK_RESPONSE
    assert citations == []
