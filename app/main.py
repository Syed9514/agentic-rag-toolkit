from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import settings
from app.embeddings import get_embeddings


def iter_supported_files(base_path: str | Path) -> list[Path]:
    root = Path(base_path)
    if not root.exists():
        return []

    allowed_suffixes = {".txt", ".md", ".csv", ".json", ".html", ".htm"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed_suffixes]


def load_documents_from_paths(paths: list[str]) -> list[Document]:
    docs: list[Document] = []
    seen: set[str] = set()

    for raw_path in paths:
        target = Path(raw_path)
        if target.is_file():
            file_paths = [target]
        elif target.is_dir():
            file_paths = iter_supported_files(target)
        else:
            continue

        for file_path in file_paths:
            if str(file_path) in seen:
                continue
            seen.add(str(file_path))

            try:
                text = file_path.read_text(encoding="utf-8")
                docs.append(Document(page_content=text, metadata={"source": str(file_path)}))
            except Exception:
                continue

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


class VectorStoreManager:
    def __init__(self, index_path: str | Path | None = None):
        self.index_path = Path(index_path or settings.vectorstore_path)
        self.embeddings = get_embeddings()
        self.index = None
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)

    def ingest_paths(self, paths: list[str], clear_existing: bool = True) -> int:
        documents = load_documents_from_paths(paths)
        if not documents:
            return 0

        split_docs = split_documents(documents)
        if clear_existing or self.index is None:
            self.index = FAISS.from_documents(split_docs, self.embeddings)
        else:
            self.index.add_documents(split_docs)

        self.index.save_local(str(self.index_path))
        return len(split_docs)

    def load_or_create(self) -> None:
        index_file = self.index_path / "index.faiss"
        if index_file.exists():
            self.index = FAISS.load_local(str(self.index_path), self.embeddings, allow_dangerous_deserialization=True)
            return

        self.index = FAISS.from_documents([], self.embeddings)

    def similarity_search(self, question: str, top_k: int = 5):
        if self.index is None:
            self.load_or_create()
        return self.index.similarity_search(question, k=top_k)
