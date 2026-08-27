from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    Wrapper around SentenceTransformers for generating semantic text embeddings.

    Concept:
    Embeddings transform textual sentences into numerical vectors (e.g. 384-dimensional arrays).
    Sentences with similar semantic meanings will be close to each other in this vector space,
    enabling math-based similarity search (e.g. Cosine distance).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Lazy loading or eager initialization of sentence-transformers model
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of text chunks.

        Args:
            texts (List[str]): List of plain text snippets.

        Returns:
            List[List[float]]: Nested list of float vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generates a vector embedding for a user query string.

        Args:
            query (str): The search prompt/question string.

        Returns:
            List[float]: Single vector embedding.
        """
        embedding = self.model.encode(query, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()
