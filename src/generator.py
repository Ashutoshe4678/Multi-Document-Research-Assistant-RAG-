import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

FALLBACK_RESPONSE = "I couldn't find enough information in the uploaded documents to answer this question."

# Active supported models on Groq API
GROQ_MODELS_TO_TRY = [
    "groq/compound",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b"
]


class RAGGenerator:
    """
    Generates grounded answers via Groq API (100% Free) or OpenAI LLM using retrieved context snippets.

    Concept:
    Grounding ensures that the LLM only synthesizes answers from the provided context.
    If the context does not contain facts needed to answer the question, the prompt instructs
    the model to return a standardized fallback message, preventing hallucination.
    """

    def __init__(self, default_model: str = "groq/compound"):
        self.default_model = default_model

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        api_key: str = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generates an answer based strictly on retrieved chunks.

        Args:
            query (str): User question.
            context_chunks (List[Dict[str, Any]]): Retrieved relevant text snippets.
            api_key (str, optional): Override API key.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: (Generated Answer, Citations List)
        """
        if not context_chunks:
            return FALLBACK_RESPONSE, []

        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = api_key or os.getenv("OPENAI_API_KEY")

        # Automatically detect if Groq key or OpenAI key is provided
        use_groq = False
        if groq_key and groq_key.strip() and not groq_key.startswith("your_"):
            use_groq = True
        elif openai_key and openai_key.startswith("gsk_"):
            groq_key = openai_key
            use_groq = True

        if not use_groq and (not openai_key or openai_key.strip() == "" or "your_" in openai_key):
            return (
                "⚠️ **API Key Missing**: Please set `GROQ_API_KEY=gsk_...` (Free at console.groq.com) or `OPENAI_API_KEY=sk-...` in your `.env` file.",
                []
            )

        # Build context block and collect citations
        context_text = ""
        citations = []
        seen_citations = set()

        for idx, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "Unknown PDF")
            page = chunk.get("page", 1)
            text = chunk.get("text", "")

            context_text += f"--- Snippet {idx} [Source: {source}, Page: {page}] ---\n{text}\n\n"

            citation_key = (source, page)
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                citations.append({"source": source, "page": page})

        system_prompt = (
            "You are a strict Multi-Document Research Assistant.\n"
            "Your task is to answer the user's question based ONLY on the provided context snippets below.\n\n"
            "STRICT RULES:\n"
            "1. Rely ONLY on the clear facts contained in the provided context.\n"
            "2. Do NOT use external prior knowledge or make assumptions beyond what is explicitly stated.\n"
            "3. If the provided context does not contain enough information to answer the question, reply EXACTLY with:\n"
            f"\"{FALLBACK_RESPONSE}\"\n"
            "4. Keep the answer clear, structured, and professional."
        )

        user_prompt = f"Context:\n{context_text}\nQuestion: {query}"

        try:
            if use_groq:
                from groq import Groq
                client = Groq(api_key=groq_key)

                answer = None
                last_err = None

                # Try active Groq models in fallback order
                for model_name in GROQ_MODELS_TO_TRY:
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.0
                        )
                        answer = response.choices[0].message.content.strip()
                        break
                    except Exception as e:
                        last_err = e
                        continue

                if answer is None:
                    raise last_err or Exception("Failed to query Groq models.")
            else:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                answer = response.choices[0].message.content.strip()

            if FALLBACK_RESPONSE.lower() in answer.lower():
                return FALLBACK_RESPONSE, []

            return answer, citations

        except Exception as e:
            return f"❌ **API Error**: {str(e)}", []
