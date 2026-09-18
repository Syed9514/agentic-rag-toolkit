from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.config import settings
from app.schemas import QueryResponse, SourceItem
from app.vectorstore import VectorStoreManager


class RAGService:
    def __init__(self) -> None:
        self.vector_store = VectorStoreManager(settings.vectorstore_path)
        self.vector_store.load_or_create()

    def ingest(self, paths: list[str], clear_existing: bool = True) -> dict[str, Any]:
        count = self.vector_store.ingest_paths(paths, clear_existing=clear_existing)
        return {"indexed_documents": count, "path": settings.vectorstore_path}

    def query(self, question: str, top_k: int = 5) -> QueryResponse:
        results = self.vector_store.similarity_search(question, top_k=top_k)

        if not results:
            return QueryResponse(
                answer="No documents were indexed yet. Add documents with the /api/ingest endpoint first.",
                sources=[],
                question=question,
                top_k=top_k,
            )

        context_text = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content.strip()}" for doc in results
        )
        answer = self._generate_answer(question, results, context_text)

        sources = [
            SourceItem(
                file=doc.metadata.get("source", "unknown"),
                score=0.0,
                content=doc.page_content[:500],
            )
            for doc in results
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            question=question,
            top_k=top_k,
            model=settings.openai_model if settings.llm_provider == "openai" else "local-rag-fallback",
        )

    def _generate_answer(self, question: str, results: list[Any], context: str) -> str:
        if settings.llm_provider == "openai" and settings.openai_api_key:
            client = OpenAI(api_key=settings.openai_api_key)
            system_message = (
                "You are a helpful assistant. Answer using only the provided retrieved context. "
                "If the information is missing, say so clearly. Cite the source files when possible."
            )
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or "No answer generated."

        best_snippets = [doc.page_content.strip() for doc in results[:3]]
        combined = "\n\n".join(best_snippets)
        return (
            "Based on the retrieved documents, here is the best grounded answer:\n\n"
            f"{combined[:1800]}\n\n"
            "Note: OpenAI API is not configured, so this is a local retrieval-based fallback answer."
        )


service = RAGService()
