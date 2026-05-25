import os
from typing import Optional

# from langchain.agents import create_agent
from langgraph.graph import StateGraph
from langchain_qwq import ChatQwen
from langgraph.utils.config import get_config
from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
import torch
from langmem import (
    create_memory_store_manager,
)

from dotenv import load_dotenv

load_dotenv()


async def main():

    class UserProfile(BaseModel):
        name: Optional[str] = None
        language: Optional[str] = None
        timezone: Optional[str] = None

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

    # memory manager
    memory_manager = create_memory_store_manager(
        model,
        store=store,
        namespace=("users", "{user_id}", "profile"),
        schemas=[UserProfile],
        enable_inserts=False,  # 会更新已经存在的profile
    )

    async def chat(messages: list):
        # 获取用户的profile
        configurable = get_config()["configurable"]
        results = store.search(("users", configurable["user_id"], "profile"))
        profile = None
        if results:
            profile = f"""<User Profile>:

    {results[0].value}
    </User Profile>
    """

        response = model.invoke(
            [
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant.{profile}""",
                },
                *messages,
            ]
        )

        # 更新profile的信息
        memory_manager.invoke({"messages": messages})
        return response

    builder = StateGraph(state_schema=str)
    builder.add_node("chat", chat)
    builder.set_entry_point("chat")
    graph = builder.compile(store=store)

    response = await graph.ainvoke(
        "我我叫xxx,来自中国", config={"configurable": {"user_id": "user-123"}}
    )
    print(response)

    response = await graph.ainvoke(
        "我刚刚通过了N1考试", config={"configurable": {"user_id": "user-123"}}
    )
    print(response)

    print("=" * 20)
    print(store.search(("users", "user-123", "profile")))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
