# Land_GPT — 多智能体深度研究系统

一套多智能体协作研究系统，核心机制包括**结构化通信协议**、**基于 Embedding 向量的非文本状态传递**和**跨任务共享记忆复用**。面向土地储备研究、政策解读、区域供需分析等复杂知识工作场景。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      接入层                              │
│          FastAPI REST + SSE 流式接口                      │
├─────────────────────────────────────────────────────────┤
│                      编排层                              │
│   DynamicOrchestrator → DAGScheduler → StateMachine      │
│   CapabilityRegistry（Agent 能力发现与握手）               │
├─────────────────────────────────────────────────────────┤
│                     智能体层                             │
│   Planner │ PolicyResearcher │ ParcelAnalyst             │
│   SupplyPlanner │ CostEstimator │ ToolAgent (CodeAct)    │
│           全部基于 ReAct（推理+行动）基类                    │
├─────────────────────────────────────────────────────────┤
│                    基础设施层                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│   │ 协作通信  │  │ 记忆系统  │  │ 对抗审查  │              │
│   │ 子系统   │  │ 子系统   │  │ 子系统   │              │
│   │+EdgeRouter│  │短期+长期 │  │红蓝对抗  │              │
│   │+VectorPool│  │+ChromaDB │  │ 收敛检测 │              │
│   └──────────┘  └──────────┘  └──────────┘              │
│   ┌──────────┐  ┌──────────┐                            │
│   │上下文压缩 │  │ CodeAct  │                            │
│   │四级流水线 │  │  沙箱    │                            │
│   └──────────┘  └──────────┘                            │
├─────────────────────────────────────────────────────────┤
│                      评测层                              │
│   规则指标 + LLM 法官评分 + Bootstrap 统计检验             │
└─────────────────────────────────────────────────────────┘
```

## 核心特性

- **6 个专业 Agent** — Planner（规划）、PolicyResearcher（政策研究）、ParcelAnalyst（地块分析）、SupplyPlanner（供应计划）、CostEstimator（成本估算）、ToolAgent（工具执行），均采用两阶段 ReAct 循环：阶段一用简洁 JSON 做行动决策，阶段二用自由文本输出完整分析，从根本上消除"长分析塞入 JSON 字段导致格式崩溃"的工程痛点

- **三种协作模式**
  - `text` — 基线模式：自然语言全文透传，信息完整但 Token 开销高
  - `structured` — 结构化模式：仅传递高密度语义单元（结论摘要 + 结构化引用），丢弃冗余推理过程，Token 节省约 50%-60%
  - `vector` — 向量增强模式：检索证据以 Embedding 向量发布到共享池，下游 Agent 按需语义取用，Token 节省约 65%-75%

- **边感知路由** — EdgeRouter 将 DAG 中 Agent 间的数据依赖边分类为 PRODUCT（分析产出，走文本通道）和 EVIDENCE（检索证据，走向量池通道），对不同类型的边采用最优传递机制

- **非文本状态传递** — SharedVectorPool 共享证据向量池，Retriever Agent 将检索到的文档块经 BailianEmbedder（阿里云百炼 text-embedding-v4，1024 维）编码后发布入池，下游 Agent 以自己的子任务目标为查询文本，通过余弦相似度从池中按需取用 top-k 条最相关证据。不同 Agent 取不同子集，实现信息分发从"广播"到"按需单播"的精细化演进

- **双层共享记忆** — 短期工作记忆（任务内共享，任务结束清空）+ 长期语义记忆（ChromaDB 持久化），支持语义检索、关键词检索、标签检索三种召回方式。任务结束时自动固化（脱敏 → 编码 → 写入 ChromaDB），新任务启动时自动召回历史经验注入 Agent 提示词

- **红蓝对抗幻觉抑制** — RedAgent 从事实准确性、数字正确性、法条引用三个维度对报告逐项攻击，BlueAgent 据证逐项修订。循环通过收敛检测（攻击数降至阈值以下即终止）和振荡检测（攻击数来回波动时终止）自动控制

- **四级上下文压缩** — 保护（法条名称/具体数字标记保护）→ 去重（跨 Agent 语义去重）→ 重筛选（按报告主题重新打分）→ 抽取式压缩（仅超 Token 预算时触发）。整条流水线每步有独立统计指标

- **DAG 并发调度** — Agent 执行建模为有向无环图，无依赖节点完全并行执行（最大并发度 4），支持全局超时、单节点超时重试和降级兜底（失败节点自动回退基座模型补写）

- **CodeAct 沙箱** — LLM 生成 Python 代码，在隔离子进程中执行（CPU ≤5s，内存 ≤512MB），实现"生成代码—隔离执行—回传结构化结果"的安全计算闭环

- **动态任务编排** — DynamicOrchestrator 通过 LLM 将用户问题自动分类为 5 种研究类型（具体地块可行性、纯政策解读、区域供需分析、类型成本对比、片区组合优化），按模板自动组装 Agent DAG，无需手动配置

- **内置评测框架** — 5 配置消融实验（基线 / 仅 DAG / 仅对抗 / 仅压缩 / 完整流水线），覆盖规则指标（事实准确率、幻觉率、引用覆盖率）和 LLM 法官评分（完整性、准确性、可溯性、连贯性、可操作性），含 Bootstrap 重采样与 Cohen's d 效应量统计检验

## 快速开始

### 环境要求

- Python ≥ 3.11
- ChromaDB（嵌入式模式，无需外部服务）
- LLM API（兼容 OpenAI 接口）

### 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd <repo-directory>

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key
```

### 环境变量

在 `agent-service/` 目录下创建 `.env` 文件：

```bash
# LLM API（DeepSeek，兼容 OpenAI 接口）
CLOUD_API_KEY=sk-your-api-key
CLOUD_BASE_URL=https://api.deepseek.com/v1
CLOUD_MODEL=deepseek-chat
CLOUD_LIGHT_MODEL=deepseek-chat

# Embedding API（阿里云百炼）
DASHSCOPE_API_KEY=sk-your-dashscope-key
BAILIAN_EMB_MODEL=text-embedding-v4
BAILIAN_EMB_DIM=1024

# ChromaDB 存储路径
CHROMA_PATH=./chroma_db

# CodeAct 沙箱超时（秒）
SANDBOX_TIMEOUT=8
```

### 初始化知识库

```bash
cd agent-service
python scripts/init_db.py
```

### 启动服务

```bash
cd agent-service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问 `http://localhost:8000/docs` 查看 Swagger 交互式 API 文档。

## API

### 提交研究任务

```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "评估余杭区某工业地块的储备方案与成本",
    "parcel_id": "YH-2024-001",
    "region": "余杭区",
    "max_rounds": 4,
    "enable_adversarial": true,
    "enable_compression": true,
    "collab_mode": "structured",
    "auto_route": false
  }'
```

### 流式接口（SSE）

```bash
curl -X POST http://localhost:8000/api/research/stream \
  -H "Content-Type: application/json" \
  -d '{"topic": "解读最新产业用地准入政策", "collab_mode": "structured"}'
```

事件格式：`status`（阶段提示）→ `delta`（报告文本增量，打字机效果）→ `done`（完整报告 + 引用 + 元数据）

### 协作模式对比

`collab_mode` 参数控制 Agent 间通信方式：

| 模式 | 说明 | Token 开销 |
|------|------|-----------|
| `text` | 自然语言全文透传（基线） | 高 |
| `structured` | 结构化紧凑协议（结论 + 引用） | 降低约 50%-60% |
| `vector` | 证据走向量池 + 结构化文本 | 降低约 65%-75% |

设置 `auto_route: true` 可由 ModeRouter 根据任务负载特征自动选择最优模式。

### 知识库管理

```bash
# 查看文档列表
curl http://localhost:8000/api/docs

# 上传文档
curl -X POST http://localhost:8000/api/docs/upload \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "policy_2024.txt", "text": "政策文件全文..."}'

# 查看文档内容
curl http://localhost:8000/api/docs/policy_2024.txt/content

# 删除文档
curl -X DELETE http://localhost:8000/api/docs/policy_2024.txt
```

## 项目结构

```
agent-service/
├── app/
│   ├── main.py                    # FastAPI 服务入口
│   ├── schemas.py                 # 全系统 Pydantic 数据模型
│   ├── agents/                    # 智能体实现
│   │   ├── base.py                # ReAct 基类（两阶段执行）
│   │   ├── planner.py             # 任务规划 Agent
│   │   ├── policy_researcher.py   # 政策研究 Agent
│   │   ├── parcel_analyst.py      # 地块分析 Agent
│   │   ├── supply_planner.py      # 供应计划 Agent
│   │   ├── cost_estimator.py      # 成本估算 Agent
│   │   └── tool_agent.py          # 工具执行 Agent（CodeAct）
│   ├── orchestrator/              # 编排引擎
│   │   ├── scheduler.py           # ResearchScheduler 总调度器
│   │   ├── dag.py                 # DAG 并发执行器
│   │   ├── dynamic_orchestrator.py # LLM 分类 + 动态编排
│   │   └── state_machine.py       # 任务状态机
│   ├── collab/                    # 协作通信子系统
│   │   ├── protocol.py            # CollabMessage 协议 + 模式渲染
│   │   ├── registry.py            # 能力注册中心（发现 + 握手）
│   │   ├── vector_pool.py         # 共享证据向量池
│   │   ├── edge_router.py         # 边类型路由（PRODUCT/EVIDENCE）
│   │   ├── mode_router.py         # 自动模式选择
│   │   └── instrument.py          # 通信埋点与指标统计
│   ├── memory/                    # 记忆子系统
│   │   ├── schema.py              # MemoryUnit 统一记忆格式
│   │   ├── layered.py             # 短期 + 长期双层记忆管理
│   │   ├── backends.py            # 后端抽象接口 + ChromaDB 实现
│   │   ├── store.py               # 记忆存储操作
│   │   └── security.py            # 数据脱敏 + 访问控制
│   ├── adversarial/               # 对抗审查子系统
│   │   ├── loop.py                # 红蓝对抗循环
│   │   ├── red_agent.py           # 红方攻击 Agent
│   │   ├── blue_agent.py          # 蓝方修订 Agent
│   │   ├── convergence.py         # 收敛/振荡检测
│   │   └── json_fallback.py       # JSON 解析容错
│   ├── compression/               # 上下文压缩子系统
│   │   ├── pipeline.py            # 保护 → 去重 → 重筛选 → 压缩
│   │   ├── compressor.py          # 抽取式压缩
│   │   ├── deduplicator.py        # 跨 Agent 语义去重
│   │   ├── protector.py           # 敏感内容保护标记
│   │   └── relevance_refilter.py  # 主题相关重筛选
│   ├── local/                     # 本地基础设施
│   │   ├── provider.py            # LocalProvider（全 Python 实现）
│   │   ├── embedding.py           # 百炼 Embedding 接入
│   │   ├── retrieval_store.py     # ChromaDB 检索存储
│   │   └── sandbox.py             # CodeAct 子进程隔离沙箱
│   ├── eval/                      # 评测层
│   │   ├── bench.py               # 5 配置消融实验运行器
│   │   ├── metrics.py             # 规则指标（准确率/幻觉率/引用覆盖）
│   │   ├── judge.py               # LLM 法官五维评分
│   │   └── stats.py               # Bootstrap + Cohen's d 统计
│   └── clients/                   # 外部服务客户端
│       ├── llm_client.py          # OpenAI 兼容 LLM 客户端
│       └── java_client.py         # HTTP 客户端（可扩展）
├── scripts/                       # 工具脚本
│   ├── init_db.py                 # 知识库初始化
│   ├── stability_test.py          # 稳定性测试（10+ 轮连续任务）
│   ├── eval_all.py                # 全量评测入口
│   └── verify_modes.py            # 三种协作模式对比验证
├── data/
│   └── docs/                      # 知识库源文档
├── pyproject.toml
└── requirements.txt
qianduan/                          # Web 前端
├── dichan_agent_xietong.dc.html   # 主页面
├── support.js                     # 前端逻辑
└── uploads/                       # 文件上传目录
```

## 评测

运行 5 配置完整消融实验：

```bash
cd agent-service
python scripts/eval_all.py
```

五种配置对比：

| 配置 | 说明 |
|------|------|
| A_baseline | 顺序执行 + 纯文本通信 |
| B_dag | DAG 并发 + 纯文本通信 |
| C_adversarial | DAG + 红蓝对抗审查 |
| D_compression | DAG + 四级上下文压缩 |
| E_full | 全部特性开启 |

评测指标包括事实准确率、幻觉率、引用覆盖率三项规则指标，以及 LLM 法官从完整性、准确性、可溯性、连贯性、可操作性五个维度给出的评分，所有对比均含 Bootstrap 统计显著性检验。

## 测试

```bash
# 稳定性测试（10+ 轮连续任务）
python scripts/stability_test.py

# 三种协作模式对比验证
python scripts/verify_modes.py
```

## 依赖

| 包 | 用途 |
|---|---|
| FastAPI + Uvicorn | 异步 Web 框架 |
| Pydantic | 数据校验与模型定义 |
| OpenAI | LLM 客户端（兼容 OpenAI 接口） |
| ChromaDB | 向量数据库（记忆存储 + 检索） |
| NumPy + SciPy + scikit-learn | 科学计算与相似度检索 |
| NetworkX | 图算法（DAG 校验） |
| HTTPX | 异步 HTTP 客户端 |

详见 `requirements.txt`。

## 设计原则

- **可插拔后端** — 记忆与检索均采用抽象接口；当前默认 ChromaDB 嵌入式向量库，可替换为 Redis + RediSearch 或 PostgreSQL + pgvector 而不改上层代码
- **优雅降级** — Agent 节点失败自动降级，空输出回退基座模型直接作答，确保不返回空白响应
- **全程可观测** — 每次 Agent 间通信事件均被埋点记录（消息次数、Token 开销、向量传输量、记忆命中率），支持 TEXT / STRUCTURED / VECTOR 三种模式的同口径定量对比
- **全 Python 技术栈** — 后端完全基于 Python + 嵌入式 ChromaDB 运行，无需 Java 环境，无需外部数据库

## 许可证

MIT
