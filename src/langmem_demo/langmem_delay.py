import os
import time

from langchain.messages import HumanMessage, AIMessage

# from langchain.agents import create_agent
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_qwq import ChatQwen
from langchain_core.prompts import ChatPromptTemplate
from langgraph.utils.config import get_store
from langgraph.store.memory import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langmem import (
    ReflectionExecutor,
    create_memory_store_manager,
)

from dotenv import load_dotenv

load_dotenv()


async def main():

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
        model, store=store, namespace=("memories",)
    )

    # executor
    executor = ReflectionExecutor(memory_manager, store=store)

    async def chat(message: str):
        response = model.invoke(message)

        to_process = {"messages": [HumanMessage(content=message)] + [response]}
        # TODO: 这个delay并不准
        delay = 0.5
        executor.submit(to_process, after_seconds=delay)
        # await memory_manager.ainvoke(to_process)
        return response.content

    builder = StateGraph(state_schema=str)
    builder.add_node("chat", chat)
    builder.set_entry_point("chat")
    graph = builder.compile(store=store)

    response = await graph.ainvoke("我喜欢猫，我的猫叫果冻")
    print(response)

    # 提取记忆
    print("第一次".center(20, "-"))
    print(store.search(("memories",)))

    time.sleep(3)
    print("第二次".center(20, "-"))
    print(store.search(("memories",)))

    executor.shutdown()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
