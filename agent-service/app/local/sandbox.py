from __future__ import annotations

import os
import sys
import json
import asyncio
import logging
import tempfile

logger = logging.getLogger(__name__)

SANDBOX_TIMEOUT = float(os.getenv("SANDBOX_TIMEOUT", "8"))

_RUNNER = r'''
import sys, json, resource
# 资源限制:CPU 时间 5s,地址空间 512MB
try:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
except Exception:
    pass

code = sys.stdin.read()
sandbox_globals = {"__builtins__": __builtins__, "result": None}
try:
    exec(code, sandbox_globals)
    out = sandbox_globals.get("result")
    print("__SANDBOX_OK__" + json.dumps({"result": out}, ensure_ascii=False, default=str))
except Exception as e:
    print("__SANDBOX_ERR__" + json.dumps({"error": str(e)}, ensure_ascii=False))
'''


class CodeActSandbox:
    """在隔离子进程中执行 LLM 生成的 Python 代码。"""

    async def run(self, code: str) -> dict:
        """执行代码,约定代码把结果写入名为 result 的变量。

        返回 {"success": bool, "result": ..., "error": ...}
        """
        with tempfile.NamedTemporaryFile(
            "w", suffix="_runner.py", delete=False, encoding="utf-8"
        ) as f:
            f.write(_RUNNER)
            runner_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                runner_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(code.encode("utf-8")), timeout=SANDBOX_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {"success": False, "result": None, "error": "sandbox timeout"}

            out = stdout.decode("utf-8", errors="replace").strip()
            for line in out.splitlines():
                if line.startswith("__SANDBOX_OK__"):
                    data = json.loads(line[len("__SANDBOX_OK__") :])
                    return {"success": True, "result": data["result"], "error": None}
                if line.startswith("__SANDBOX_ERR__"):
                    data = json.loads(line[len("__SANDBOX_ERR__") :])
                    return {"success": False, "result": None, "error": data["error"]}
            return {
                "success": False,
                "result": None,
                "error": f"no sandbox marker; stderr={stderr.decode('utf-8', 'replace')[:200]}",
            }
        finally:
            try:
                os.unlink(runner_path)
            except OSError:
                pass
