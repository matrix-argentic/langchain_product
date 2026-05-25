import os
from typing import Optional

# from langchain.agents import create_agent
from langgraph.graph import StateGraph
from langchain_qwq import ChatQwen
from langgraph.utils.config import get_config
from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
import torch
from langmem import (
    create_memory_store_manager,
)

from dotenv import load_dotenv

load_dotenv()


async def main():

    # 1. vector store
    # embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    # 归一化后，嵌入向量的长度为1，这可以提高相似度计算的准确性
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = os.getenv("EMBEDDING_MODEL")
    hf_home = os.getenv("HF_HOME")
    model_cache_dir = hf_home + "\\hub"
    embed = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        # show_progress=True,
        cache_folder=model_cache_dir,  # 缓存模型文件, 避免重复下载
    )
    store = InMemoryStore(
        index={
            "dims": 1024,
            "embed": embed,
        }
    )

    # 2. memory manager with store

    class Episode(BaseModel):
        """以代理（Agent）的第一人称视角撰写事件记录。利用事后回顾（hindsight）来记录记忆，保存代理关键的思考过程，以便其随着时间推移不断学习。"""

        observation: str = Field(..., description="上下文与背景设定——发生了什么")
        thoughts: str = Field(
            ...,
            description='代理在该事件中得出正确行动和结果所依据的内部推理过程与观察。 使用 "我……" 的口吻。',
        )
        action: str = Field(
            ...,
            description='采取了什么行动、如何执行以及采用何种格式。（需包含对行动成功至关重要的要素）。使用 "我……" 的口吻。',
        )
        result: str = Field(
            ...,
            description='结果与回顾反思。做得好的地方是什么？下次如何改进？使用 "我……" 的口吻。',
        )

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

    # memory manager
    memory_manager = create_memory_store_manager(
        model,
        store=store,
        namespace=(
            "memories",
            "episodes",
        ),
        schemas=[Episode],
        instructions="提炼出值得关注的卓越问题解决案例，并阐明其行之有效的原因。",
        enable_inserts=True,
    )

    async def chat(messages: list):
        similar = store.search(
            ("memories", "episodes"),
            query=messages[-1]["content"],
            limit=1,
        )
        system_message = "你是一个助手."
        if similar:
            system_message += "\n\n### 情景记忆:"
            for i, item in enumerate(similar, start=1):
                episode = item.value["content"]
                system_message += f"""
        Episode {i}:
        When: {episode['observation']}
        Thought: {episode['thoughts']}
        Did: {episode['action']}
        Result: {episode['result']}
                """

        # 使用情景记忆回答问题
        response = model.invoke(
            [{"role": "system", "content": system_message}, *messages]
        )

        # 存储情景记忆
        memory_manager.invoke({"messages": messages})
        return response

    builder = StateGraph(state_schema=list)
    builder.add_node("chat", chat)
    builder.set_entry_point("chat")
    graph = builder.compile(store=store)

    response = await graph.ainvoke(
        [{"role": "user", "content": "什么是二叉树？"}],
        config={"configurable": {"user_id": "user-123"}},
    )
    print(response)

    print(store.search(("memories", "episodes"), query="二叉树"))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
