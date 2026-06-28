import asyncio
import os
from pathlib import Path


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _load_env() -> None:
    """加载 .env，确保 DASHSCOPE_API_KEY 等被读进环境变量。"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except Exception:
        pass
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

from app.local.provider import LocalProvider  # noqa: E402


async def main():
    provider = LocalProvider()
    n = await provider.store.ingest_dir("./data/docs")
    print(f"✓ 已入库 {n} 个文档切片到 ./chroma_db")
    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
