from typing import List, Dict, Any
import pymupdf  # Modern PyMuPDF API


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from PDF file bytes using PyMuPDF.

    Concept:
    PyMuPDF reads the internal structure of the PDF file line by line.
    We iterate over each page, extract plain text, and associate it with essential
    metadata (source filename and 1-indexed page number) needed for citations.

    Args:
        file_bytes (bytes): The raw bytes of the uploaded PDF file.
        filename (str): Display name of the source PDF document.

    Returns:
        List[Dict[str, Any]]: List of objects containing 'text', 'source', and 'page'.
    """
    extracted_pages = []

    if not file_bytes:
        print(f"[Warning] Empty file bytes received for: {filename}")
        return extracted_pages

    try:
        # Open document from in-memory stream
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text").strip()

            # Skip empty pages or pages with only whitespace
            if text:
                extracted_pages.append({
                    "text": text,
                    "source": filename,
                    "page": page_idx + 1  # Human-readable 1-based indexing
                })

        doc.close()
    except Exception as e:
        print(f"[Error] Failed to parse PDF '{filename}': {str(e)}")

    return extracted_pages
