import os

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_qwq import ChatQwen
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from langmem import (
    create_manage_memory_tool,
    create_memory_store_manager,
    create_search_memory_tool,
)

from dotenv import load_dotenv

load_dotenv()


async def main():
    store = InMemoryStore()

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

    memory_manager = create_memory_store_manager(model, namespace=("memories",))

    async def chat(message: str):
        response = model.invoke(message)

        to_process = {"messages": [HumanMessage(content=message)] + [response]}
        await memory_manager.ainvoke(to_process)
        return response.content

    builder = StateGraph(state_schema=str)
    builder.add_node("chat", chat)
    builder.set_entry_point("chat")
    graph = builder.compile(store=store)

    response = await graph.ainvoke("我喜欢猫，我的猫叫果冻")
    print(response)

    # 提取记忆
    print(store.search(("memories",)))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main=main())
