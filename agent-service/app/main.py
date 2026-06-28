import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path


def _load_env() -> None:
    """加载 .env 中的密钥到环境变量。"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except Exception:
        pass
    # 退回手动解析（无需 python-dotenv 依赖）
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_env()

import json as _json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.clients.llm_client import LLMClient
from app.local.provider import LocalProvider
from app.orchestrator.scheduler import ResearchScheduler
from app.schemas import ResearchRequest, ResearchResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agent service starting up...")
    # 使用全 Python 的 LocalProvider(本地 Chroma 检索 + 百炼 embedding),
    # 不依赖 Java 后端即可独立运行。
    app.state.provider = LocalProvider()
    app.state.llm_client = LLMClient()
    logger.info("Clients initialized (LocalProvider mode)")

    yield

    logger.info("Agent service shutting down...")
    for client_name, client in [("llm_client", app.state.llm_client), ("provider", app.state.provider)]:
        try:
            await client.close()
        except Exception:
            logger.exception(f"Error closing {client_name}")
    logger.info("Clients closed")


app = FastAPI(
    title="Land_GPT Agent Orchestration Service",
    description="Multi-Agent Deep Research System — Python orchestration layer",
    version="0.1.0",
    lifespan=lifespan,
)

# 允许前端 HTML(本地文件 / 任意端口)跨域调用本服务
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-orchestrator"}


@app.post("/api/research", response_model=ResearchResponse)
async def execute_research(request: ResearchRequest, req: Request):
    """Execute a full multi-agent deep research task.

    This is the main entry point. It orchestrates:
    1. DAG-based concurrent agent execution (4 agents)
    2. Red-Blue adversarial hallucination suppression (optional)
    3. Three-level context compression (optional)
    """
    try:
        scheduler = ResearchScheduler(
            java_client=req.app.state.provider,
            llm_client=req.app.state.llm_client,
        )
        result = await scheduler.execute(request)
        return result
    except Exception as e:
        logger.exception("Research execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research/stream")
async def execute_research_stream(request: ResearchRequest, req: Request):
    """流式版研究接口(SSE)。

    先完成多 Agent 研究,再把最终报告**逐块流式推送**(打字机式),
    最后推送一条 done 事件携带引用与 metadata。前端用 EventSource/
    fetch-stream 接收,实现报告逐字呈现。

    事件格式(SSE, 每条以 data: 开头):
      {"type":"status","msg":"..."}      阶段提示
      {"type":"delta","text":"..."}      报告增量文本
      {"type":"done","citations":[...],"metadata":{...}}  结束
      {"type":"error","detail":"..."}    出错
    """
    import asyncio

    async def event_gen():
        try:
            yield _sse({"type": "status", "msg": "多 Agent 协作研究中…"})
            scheduler = ResearchScheduler(
                java_client=req.app.state.provider,
                llm_client=req.app.state.llm_client,
            )
            result = await scheduler.execute(request)

            yield _sse({"type": "status", "msg": "生成报告…"})
            report = result.report or ""
            # 逐块流式推送(按句/换行切,兼顾打字机观感与效率)
            buf = ""
            for ch in report:
                buf += ch
                if len(buf) >= 8 or ch in "\n。！？;:，、":
                    yield _sse({"type": "delta", "text": buf})
                    buf = ""
                    await asyncio.sleep(0.012)  # 控制打字机速度
            if buf:
                yield _sse({"type": "delta", "text": buf})

            yield _sse({
                "type": "done",
                "task_id": result.task_id,
                "citations": result.citations,
                "agent_results": result.agent_results,
                "metadata": result.metadata,
            })
        except Exception as e:
            logger.exception("Stream research failed")
            yield _sse({"type": "error", "detail": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(obj: dict) -> str:
    """把一个 dict 序列化为一条 SSE 消息。"""
    return "data: " + _json.dumps(obj, ensure_ascii=False) + "\n\n"


# ── 知识库管理接口（真实增删查）────────────────────────────────
@app.get("/api/docs")
async def list_docs(req: Request):
    """列出知识库里真实存在的文档（按 doc_id 聚合）。"""
    try:
        store = req.app.state.provider.store
        docs = store.list_documents()
        return {"docs": docs, "doc_count": len(docs),
                "chunk_count": sum(d["chunks"] for d in docs)}
    except Exception as e:
        logger.exception("List docs failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/docs/upload")
async def upload_doc(req: Request):
    """上传一个文档到知识库并真实入库（切块+向量化）。

    请求体 JSON: {"doc_id": "文件名", "text": "文档全文"}
    （前端读取用户选择的 .txt/.md 文件内容后提交）
    """
    try:
        body = await req.json()
        doc_id = (body.get("doc_id") or "").strip()
        text = body.get("text") or ""
        if not doc_id or not text.strip():
            raise HTTPException(status_code=400, detail="doc_id 和 text 不能为空")
        store = req.app.state.provider.store
        n = await store.ingest_text(doc_id, text)
        return {"ok": True, "doc_id": doc_id, "chunks": n}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload doc failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/docs/{doc_id}")
async def delete_doc(doc_id: str, req: Request):
    """从知识库真实删除某文档的所有块。"""
    try:
        store = req.app.state.provider.store
        n = store.delete_document(doc_id)
        return {"ok": True, "doc_id": doc_id, "deleted_chunks": n}
    except Exception as e:
        logger.exception("Delete doc failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/docs/{doc_id}/content")
async def view_doc(doc_id: str, req: Request):
    """查看某文档的全文内容(用于前端展开查看)。"""
    try:
        store = req.app.state.provider.store
        return store.get_document(doc_id)
    except Exception as e:
        logger.exception("View doc failed")
        raise HTTPException(status_code=500, detail=str(e))
