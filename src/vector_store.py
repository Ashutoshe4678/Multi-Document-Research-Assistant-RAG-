import os
import chromadb
from typing import List, Dict, Any, Optional
from src.embeddings import EmbeddingManager


class VectorStoreManager:
    """
    Manages persistent local ChromaDB vector database storage and queries.

    Concept:
    ChromaDB stores document embeddings, text content, and structured metadata.
    It builds an indexing structure (HNSW) allowing lightning-fast k-Nearest-Neighbors (k-NN)
    similarity lookups between a query vector and chunk vectors.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "pdf_rag_collection",
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager or EmbeddingManager()

        # Create persistence directory if missing
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize ChromaDB Client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Computes embeddings for chunks and adds them to ChromaDB.
        Deduplicates chunks using their deterministic chunk IDs.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk items with id, text, and metadata.

        Returns:
            int: Total count of new chunks added.
        """
        if not chunks:
            return 0

        # Retrieve existing IDs to prevent duplicate ingestion
        existing_records = self.collection.get()
        existing_ids = set(existing_records["ids"]) if existing_records and "ids" in existing_records else set()

        new_chunks = [c for c in chunks if c["id"] not in existing_ids]

        if not new_chunks:
            return 0

        ids = [c["id"] for c in new_chunks]
        documents = [c["text"] for c in new_chunks]
        metadatas = [c["metadata"] for c in new_chunks]

        # Generate vectors
        embeddings = self.embedding_manager.embed_texts(documents)

        # Upsert into collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(new_chunks)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against stored document chunks.

        Args:
            query_text (str): User question.
            top_k (int): Number of most relevant context chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: List of retrieved items with text, source, page, distance.
        """
        if self.collection.count() == 0:
            return []

        # Embed query text into 384-d vector
        query_embedding = self.embedding_manager.embed_query(query_text)

        # Limit top_k to total stored items
        k = min(top_k, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                retrieved_chunks.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown PDF"),
                    "page": meta.get("page", 1),
                    "distance": round(float(dist), 4)
                })

        return retrieved_chunks

    def clear_store(self):
        """Removes all stored vectors and documents from ChromaDB."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_count(self) -> int:
        """Returns total number of chunks currently stored in vector DB."""
        return self.collection.count()
