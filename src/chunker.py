from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    documents: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Splits extracted document pages into smaller semantic chunks while keeping metadata.

    Concept:
    Vector similarity search works best when text segments are concise and self-contained.
    RecursiveCharacterTextSplitter splits text recursively on common separators
    (paragraphs '\\n\\n', lines '\\n', spaces ' ') to keep related sentences together.

    Args:
        documents (List[Dict[str, Any]]): List of extracted page dicts (text, source, page).
        chunk_size (int): Target maximum character count per chunk.
        chunk_overlap (int): Number of overlapping characters between adjacent chunks.

    Returns:
        List[Dict[str, Any]]: List of chunk dicts with 'id', 'text', and 'metadata'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for doc in documents:
        page_text = doc.get("text", "")
        source = doc.get("source", "unknown")
        page = doc.get("page", 1)

        if not page_text:
            continue

        # Split current page text into chunks
        split_snippets = splitter.split_text(page_text)

        for chunk_idx, snippet in enumerate(split_snippets):
            # Create a unique string key for deduplication
            chunk_id = f"{source}_p{page}_c{chunk_idx}"

            chunks.append({
                "id": chunk_id,
                "text": snippet,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_idx
                }
            })

    return chunks
