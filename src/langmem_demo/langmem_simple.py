import os

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_qwq import ChatQwen
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
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
    # create_manage_memory_tool 的 agent的提示一部分
    """
    当您出现以下情况时，请主动调用此工具：
    1.识别出用户的新偏好。
    2.收到用户明确提出的记忆某项内容或更改行为方式的请求。
    3.正在处理任务并希望记录重要上下文。
    4.发现现有的记忆内容不正确或过时。
    """
    # create_search_memory_tool 的 agent的提示一部分
    """
    根据当前的上下文，检索你长期记忆中相关的信息。
    """

    agent = create_agent(
        model=model,
        tools=[
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",)),
        ],
        store=store,
    )

    resp = agent.invoke(
        input={"messages": [HumanMessage(content="记住我喜欢深色模式")]}
    )

    print(resp["messages"][-1].content)

    resp = agent.invoke(input={"messages": [HumanMessage(content="我喜欢什么模式?")]})

    print(resp["messages"][-1].content)
