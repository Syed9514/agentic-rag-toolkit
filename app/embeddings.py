from __future__ import annotations

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import settings


def get_embeddings():
    if settings.embedding_provider == "openai" and settings.openai_api_key:
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
