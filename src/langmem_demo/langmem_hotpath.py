import os

from langchain.messages import HumanMessage, AIMessage

# from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from langchain_qwq import ChatQwen
from langchain_core.prompts import ChatPromptTemplate
from langgraph.utils.config import get_store
from langgraph.store.memory import InMemoryStore
from langmem import (
    create_manage_memory_tool,
)

from dotenv import load_dotenv

load_dotenv()


async def main():
    # 我们自己search，而不是交给工具search
    def prompt(state):
        store = get_store()
        # query 支持自然语言搜索，历史，也就是说可以存储到向量数据库中，或者postgres这种带向量存储的
        memories = store.search(("memories",), query=state["messages"][-1].content)
        system_msg = f"""你是一个助手。
    ## 记忆
    <memories>
    {memories}
    </memories>
    """
        return ChatPromptTemplate.from_messages(
            messages=[AIMessage(content=system_msg), *state["messages"]]
        )

    from langchain_community.embeddings import HuggingFaceEmbeddings
    import torch

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
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()

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
    agent = create_react_agent(
        model=model,
        prompt=prompt,
        tools=[create_manage_memory_tool(namespace=("memories",))],
        store=store,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "thread-a"}}

    response = agent.invoke(
        input={"messages": [HumanMessage(content="你知道我喜欢什么动物么？")]},
        config=config,
    )
    print(response["messages"][-1].content)

    response = agent.invoke(
        input={"messages": [HumanMessage(content="记住，我喜欢猫咪")]},
        config=config,
    )
    print(response["messages"][-1].content)

    config = {"configurable": {"thread_id": "thread-b"}}
    response = agent.invoke(
        input={"messages": [HumanMessage(content="你知道我喜欢什么动物么？")]},
        config=config,
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main=main())
