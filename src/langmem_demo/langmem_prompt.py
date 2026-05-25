import os

from dotenv import load_dotenv
from langchain_qwq import ChatQwen
from langmem import create_prompt_optimizer

load_dotenv()

trajectories = [
    # 无标注的对话（仅包含对话内容）
    (
        [
            {"role": "user", "content": "给我讲讲火星"},
            {"role": "assistant", "content": "火星是太阳系第四颗行星……"},
            {"role": "user", "content": "我想了解更多关于它卫星的信息"},
        ],
        None,  # 无标注
    ),
    # 带评分与评论的对话
    (
        [
            {"role": "user", "content": "火星的卫星有哪些？"},
            {
                "role": "assistant",
                "content": "火星有两颗卫星：火卫一（Phobos）和火卫二（Deimos）……",
            },
        ],
        {
            "score": 0.9,
            "comment": "应包含更多细节，并推荐后续追问问题",
        },
    ),
    # 标注也可以是不同类型的，例如编辑/修订！
    (
        [
            {"role": "user", "content": "比较一下火星和地球"},
            {"role": "assistant", "content": "火星和地球有许多不同之处……"},
        ],
        {"revised": "地球和火星在许多方面既有相似之处也有差异……"},
    ),
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


optimizer = create_prompt_optimizer(
    model,
    kind="metaprompt",
    config={"max_reflection_steps": 1, "min_reflection_steps": 0},
)

updated = optimizer.invoke(
    {"trajectories": trajectories, "prompt": "你是一位行星科学专家。"}
)

print(updated)
