from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryBackend(ABC):
    """长期记忆存储后端抽象。任何实现此接口的后端都可热插拔。"""

    @abstractmethod
    async def upsert(self, mem_id: str, vector: list[float], document: str, metadata: dict) -> None:
        """写入/更新一条记忆。"""

    @abstractmethod
    async def query(self, vector: list[float], top_k: int) -> list[dict]:
        """语义检索,返回 [{id, document, metadata, distance}, ...]。"""

    @abstractmethod
    def get_all(self) -> list[dict]:
        """取全部(用于关键词/标签过滤)。"""

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class ChromaBackend(MemoryBackend):
    """默认实现:ChromaDB 嵌入式向量库(零运维,openEuler 友好)。"""

    def __init__(self, path: str, collection: str = "shared_memory"):
        import chromadb
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._collection_name = collection

    async def upsert(self, mem_id, vector, document, metadata):
        self._col.upsert(
            ids=[mem_id], embeddings=[vector], documents=[document], metadatas=[metadata]
        )

    async def query(self, vector, top_k):
        if self._col.count() == 0:
            return []
        res = self._col.query(query_embeddings=[vector], n_results=min(top_k, self._col.count()))
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"document": doc, "metadata": meta, "distance": float(dist)})
        return out

    def get_all(self):
        data = self._col.get()
        return [
            {"document": d, "metadata": m}
            for d, m in zip(data.get("documents", []), data.get("metadatas", []))
        ]

    def count(self):
        return self._col.count()

    def clear(self):
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
