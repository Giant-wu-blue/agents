from __future__ import annotations

import os
import re
import glob
import logging
from dataclasses import dataclass

from app.local.embedding import BailianEmbedder

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./data/docs")


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    doc_id: str


def _decode_uname(name: str) -> str:
    if "#U" not in name:
        return name
    try:
        return re.sub(r"#U([0-9a-fA-F]{4})",
                      lambda m: chr(int(m.group(1), 16)), name)
    except Exception:
        return name


def _split_text(text: str, max_len: int = 300) -> list[str]:
    sentences = [s.strip() for s in re.split(r"[。！？\n]+", text) if s.strip()]
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) > max_len and buf:
            chunks.append(buf)
            buf = s
        else:
            buf = buf + s if not buf else buf + "。" + s
    if buf:
        chunks.append(buf)
    return chunks


class LocalRetrievalStore:
    """文档检索库:建库(ingest)+ 检索(search)。"""

    def __init__(self, embedder: BailianEmbedder | None = None):
        self.embedder = embedder or BailianEmbedder()
        self._client = None
        self._col = None

    def _get_collection(self):
        if self._col is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=CHROMA_PATH)
            # 用 None 作为 embedding_function:我们自己传向量进去,不让 chroma 自己算
            self._col = self._client.get_or_create_collection(
                name="kb_chunks", metadata={"hnsw:space": "cosine"}
            )
        return self._col

    async def ingest_dir(self, docs_dir: str = DOCS_DIR) -> int:
        """把 docs_dir 下所有 .txt/.md 文档切块、编码、入库。返回入库块数。"""
        col = self._get_collection()
        paths = glob.glob(os.path.join(docs_dir, "**", "*.txt"), recursive=True)
        paths += glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)

        all_chunks, all_ids, all_meta = [], [], []
        for p in paths:
            doc_id = os.path.basename(p)
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
            for j, chunk in enumerate(_split_text(text)):
                all_chunks.append(chunk)
                all_ids.append(f"{doc_id}::chunk_{j}")
                all_meta.append({"doc_id": doc_id})

        if not all_chunks:
            logger.warning(f"未在 {docs_dir} 找到任何文档,检索库为空")
            return 0

        vectors = await self.embedder.encode(all_chunks)
        col.add(ids=all_ids, embeddings=vectors, documents=all_chunks, metadatas=all_meta)
        logger.info(f"检索库建库完成:{len(all_chunks)} 块 来自 {len(paths)} 个文档")
        return len(all_chunks)

    async def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """语义检索 + 重排,返回 top_k chunk。

        流程:向量召回更多候选(top_k×3)→ gte-rerank 精排 → 取前 top_k。
        rerank 不可用时自动降级为纯向量排序。
        """
        col = self._get_collection()
        qvec = await self.embedder.encode_one(query)
        # 召回更大候选池,给 reranker 更多精选空间
        recall_k = min(max(top_k * 3, top_k), 30)
        res = col.query(query_embeddings=[qvec], n_results=recall_k)
        chunks: list[RetrievedChunk] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            chunks.append(
                RetrievedChunk(
                    id=cid,
                    text=doc,
                    score=1.0 - float(dist),  # cosine 距离转相似度
                    doc_id=meta.get("doc_id", "unknown"),
                )
            )

        # ── cross-encoder 重排(gte-rerank)──
        from app.local.reranker import rerank
        order = await rerank(query, [c.text for c in chunks], top_n=top_k)
        if order is not None:
            # 按 reranker 给出的顺序重排,并取前 top_k
            chunks = [chunks[i] for i in order if i < len(chunks)][:top_k]
        else:
            # 降级:保持向量相似度顺序,取前 top_k
            chunks = chunks[:top_k]
        return chunks

    # ── 知识库管理:列出 / 上传 / 删除（供前端真实增删查）──────────
    def list_documents(self) -> list[dict]:
        """按 doc_id 聚合，返回每个文档的块数等信息(含可读名)。"""
        col = self._get_collection()
        got = col.get(include=["metadatas"])
        ids = got.get("ids", []) or []
        metas = got.get("metadatas", []) or []
        agg: dict[str, int] = {}
        for cid, meta in zip(ids, metas):
            doc_id = (meta or {}).get("doc_id") or cid.split("::")[0]
            agg[doc_id] = agg.get(doc_id, 0) + 1
        return [{"doc_id": d, "name": _decode_uname(d), "chunks": c}
                for d, c in sorted(agg.items())]

    def get_document(self, doc_id: str) -> dict:
        """取某文档的全部切片内容(按 chunk 序号拼接),用于前端查看。"""
        col = self._get_collection()
        got = col.get(include=["documents", "metadatas"])
        ids = got.get("ids", []) or []
        docs = got.get("documents", []) or []
        metas = got.get("metadatas", []) or []
        items = []
        for cid, doc, meta in zip(ids, docs, metas):
            did = (meta or {}).get("doc_id") or cid.split("::")[0]
            if did == doc_id:
                # 取 chunk 序号用于排序
                m = cid.rsplit("::chunk_", 1)
                order = int(m[1]) if len(m) == 2 and m[1].isdigit() else 0
                items.append((order, doc))
        items.sort(key=lambda x: x[0])
        content = "\n".join(t for _, t in items)
        return {"doc_id": doc_id, "name": _decode_uname(doc_id),
                "chunks": len(items), "content": content}

    async def ingest_text(self, doc_id: str, text: str) -> int:
        """把一段文本作为一个文档切块入库，返回入库块数。
        若同名 doc_id 已存在，先删除旧的再入库（覆盖更新）。"""
        if not text.strip():
            return 0
        self.delete_document(doc_id)  # 覆盖式：先清旧块
        col = self._get_collection()
        chunks = _split_text(text)
        ids = [f"{doc_id}::chunk_{j}" for j in range(len(chunks))]
        metas = [{"doc_id": doc_id} for _ in chunks]
        vectors = await self.embedder.encode(chunks)
        col.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metas)
        logger.info(f"上传入库：{doc_id} 共 {len(chunks)} 块")
        return len(chunks)

    def delete_document(self, doc_id: str) -> int:
        """删除某文档的所有块。返回删除的块数。"""
        col = self._get_collection()
        got = col.get(include=["metadatas"])
        ids = got.get("ids", []) or []
        metas = got.get("metadatas", []) or []
        to_del = [
            cid for cid, meta in zip(ids, metas)
            if ((meta or {}).get("doc_id") or cid.split("::")[0]) == doc_id
        ]
        if to_del:
            col.delete(ids=to_del)
            logger.info(f"删除文档：{doc_id} 共 {len(to_del)} 块")
        return len(to_del)
