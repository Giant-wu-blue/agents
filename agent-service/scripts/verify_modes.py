import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.collab.protocol import upstream_to_text, upstream_to_structured
from app.collab.instrument import estimate_tokens

# 模拟一份上游 agent 产出(含完整推理过程,贴近真实)
UPSTREAM = {
    "policy": {
        "content": ("该地块符合余杭区国土空间总体规划与产业准入目录；现状用地性质为"
                    "一类工业用地（M1），与区级先进制造业导向一致，不存在政策冲突。"
                    "依据《杭州市余杭区工业用地储备管理办法》，储备需完成征地补偿、"
                    "社保占补平衡、七通一平等前期开发，建议按程序推进。" * 2),
        "citations": ["余杭国土空间规划 chunk#04", "基准地价表 chunk#09",
                      "储备管理办法 chunk#12"],
        "scratchpad": [
            "Thought: 先检索政策库确认准入要求……（一段较长的推理过程文本）" * 3,
            "Observation: 命中3条政策证据，覆盖准入、补偿、开发时序……" * 3,
            "Thought: 综合判断无政策冲突，进入结论……" * 2,
        ],
    },
    "parcel": {
        "content": ("地块为 M1 一类工业，容积率上限 2.0，周边路网与市政配套成熟，"
                    "储备适宜性评分 0.78，主要风险在拆迁周期。" * 2),
        "citations": ["地块现状表 chunk#21"],
        "scratchpad": ["Thought: 从向量池取地块条件证据……" * 3],
    },
}


def main():
    t_text = upstream_to_text(UPSTREAM)
    t_struct = upstream_to_structured(UPSTREAM)

    tok_text = estimate_tokens(t_text)
    tok_struct = estimate_tokens(t_struct)
    # 向量模式:结论部分同结构化,但证据(scratchpad/检索原文)走向量池不计文本
    # 这里以结构化为基准(向量模式文本通信 ≈ 结构化的结论部分,甚至更少)
    tok_vector = tok_struct

    print("=" * 56)
    print("三模式 Agent 间通信 token 对比(模拟数据,不调用 LLM)")
    print("=" * 56)
    print(f"  纯文本协作 (text)       : {tok_text:>5} token")
    print(f"  结构化协议 (structured) : {tok_struct:>5} token  "
          f"(省 {round((tok_text-tok_struct)/tok_text*100)}%)")
    print(f"  向量直传   (vector)     : {tok_vector:>5} token  "
          f"(证据走向量池,文本部分≈结构化)")
    print("=" * 56)
    if tok_text > tok_struct:
        print("✓ 模式区分正确:纯文本 token 明显高于结构化/向量。")
        print("  若实跑 eval_all 三模式仍接近,则说明该任务 Agent 上游")
        print("  通信量偏小(并行节点多),可换上游依赖更深的任务凸显差异。")
    else:
        print("✗ 异常:结构化未低于纯文本,请检查 protocol.py 渲染逻辑。")


if __name__ == "__main__":
    main()
