import os

from dotenv import load_dotenv
from langchain_qwq import ChatQwen
from langmem import create_multi_prompt_optimizer

load_dotenv()

# 示例团队：研究员负责查找信息，撰稿人负责生成报告
conversations = [
    (
        [
            {"role": "user", "content": "调研量子计算的最新进展"},
            {
                "role": "assistant",
                "content": "找到了几篇关于量子霸权（quantum supremacy）的论文……",
            },
            {
                "role": "assistant",
                "content": "近期的量子计算发展表明……",
            },
            {"role": "user", "content": "这份报告缺少实现细节"},
        ],
        # 未提供显式反馈，但优化器可以从对话内容中推断质量
        None,
    ),
    (
        [
            {"role": "user", "content": "分析新型机器学习模型"},
            {"role": "assistant", "content": "关于架构的主要发现：……"},
            {"role": "assistant", "content": "基于上述研究，这些模型……"},
            {"role": "user", "content": "报告很棒，非常详尽"},
        ],
        # 针对整个团队的数值评分
        {"score": 0.95},
    ),
]

prompts = [
    {
        "name": "researcher",
        "prompt": "你负责分析技术论文并提炼关键发现",
    },
    {
        "name": "writer",
        "prompt": "你负责基于研究结果撰写清晰的报告",
    },
]

api_key = os.getenv("DASHSCOPE_API_KEY", "")
api_base = os.getenv("DASHSCOPE_API_BASE", "")
model_name = os.getenv("DASHSCOPE_MODEL_NAME", "")
model = ChatQwen(
    model=model_name,
    max_tokens=3_000,
    timeout=None,
    max_retries=2,
    api_key=api_key,
    base_url=api_base,
    enable_thinking=False,  # 关闭深度思考
    # other params...
)


optimizer = create_multi_prompt_optimizer(
    model,
    kind="gradient",  # Best for team dynamics
    config={"max_reflection_steps": 3},
)

updated = optimizer.invoke({"trajectories": conversations, "prompts": prompts})

print(updated)
