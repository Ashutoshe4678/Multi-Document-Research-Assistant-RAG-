from typing import List, Dict, Any
from src.vector_store import VectorStoreManager


class RAGRetriever:
    """
    Retriever module for semantic context search.

    Concept:
    Acts as the bridge between the user's natural language question and the vector store.
    It passes the query to ChromaDB and formats the retrieved chunks with document metadata.
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant text snippets from the vector database.

        Args:
            query (str): User question.
            top_k (int): Maximum number of top matching chunks to return.

        Returns:
            List[Dict[str, Any]]: List of matching chunks with text and page citations.
        """
        if not query or not query.strip():
            return []

        return self.vector_store.query(query_text=query.strip(), top_k=top_k)
