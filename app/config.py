from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Agentic RAG Toolkit"
    api_prefix: str = "/api"
    docs_path: str = "docs"
    vectorstore_path: str = ".vectorstore"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    llm_provider: str = "openai"
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    log_level: str = "INFO"


settings = Settings()
