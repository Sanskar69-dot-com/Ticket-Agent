from __future__ import annotations

import hashlib
from pathlib import Path
import warnings

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma retrieval scaffold with stable output contracts."""

    RELEVANCE_THRESHOLD = 0.25

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    async def initialize(self) -> None:
        try:
            persist_dir = Path(self.settings.vector_db_path).resolve()
            persist_dir.mkdir(parents=True, exist_ok=True)

            raw_docs = load_support_documents(self.documents_dir)
            documents = split_support_documents(raw_docs)
            embeddings = build_embeddings(self.settings)

            # Sanitize document metadata to safe filenames only
            for doc in documents:
                doc.metadata["source"] = Path(doc.metadata.get("source", "")).name

            store = Chroma(
                collection_name=self.settings.rag_collection,
                embedding_function=embeddings,
                persist_directory=str(persist_dir),
            )

            # Avoid duplicating chunks across restarts using content hashes
            existing_data = store.get()
            existing_ids = set(existing_data["ids"]) if existing_data and "ids" in existing_data else set()

            new_docs = []
            new_ids = []
            for doc in documents:
                chunk_id = hashlib.sha256(
                    f"{doc.metadata.get('source')}:{doc.page_content}".encode("utf-8")
                ).hexdigest()
                if chunk_id not in existing_ids:
                    new_docs.append(doc)
                    new_ids.append(chunk_id)
                    existing_ids.add(chunk_id)

            if new_docs:
                store.add_documents(documents=new_docs, ids=new_ids)

            self._store = store
        except Exception as exc:
            self._store = None
            raise ComponentNotReadyError(f"Failed to initialize knowledge retriever: {exc}") from exc

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        if not query or not query.strip():
            return []

        if self._store is None:
            raise ComponentNotReadyError("KnowledgeRetriever is not initialized")

        k = limit if limit is not None else self.settings.rag_top_k

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Relevance scores must be between 0 and 1.*")
            try:
                results_with_scores = self._store.similarity_search_with_relevance_scores(
                    query.strip(), k=k
                )
            except Exception:
                docs = self._store.similarity_search(query.strip(), k=k)
                results_with_scores = [(doc, 1.0) for doc in docs]

        normalized_results: list[dict[str, str]] = []
        for doc, score in results_with_scores:
            if score >= self.RELEVANCE_THRESHOLD:
                safe_source = Path(doc.metadata.get("source", "")).name
                normalized_results.append({
                    "content": doc.page_content,
                    "source": safe_source,
                })

        return normalized_results
