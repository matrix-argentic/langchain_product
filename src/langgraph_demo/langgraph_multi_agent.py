import os
from typing import TypedDict


from dotenv import load_dotenv
from langchain_community.tools import tool
from langgraph.graph import END, StateGraph
from tavily.client import TavilyClient
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

from langchain_qwq import ChatQwen

# 准备环境变量
load_dotenv()

# 模型相关
api_key = os.getenv("DASHSCOPE_API_KEY", "")
api_base = os.getenv("DASHSCOPE_API_BASE", "")
model_name = os.getenv("DASHSCOPE_MODEL_NAME", "")

# 搜索相关
tavily_api_key = os.getenv("TAVILY_API_KEY", "")


class ResearchState(TypedDict):
    # 主题
    topic: str
    # 研究笔记
    research_notes: str
    # 草稿
    draft: str
    # 评审反馈
    review_feedback: str
    # 最终文章
    final_article: str
    # 下一步行动
    next_step: str


tavily_client = TavilyClient(api_key=tavily_api_key)

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


@tool
def search_web(query: str) -> str:
    """在网络上搜索信息"""
    try:
        resp = tavily_client.search(query=query, num_results=5)
        results = resp.get("results", [])
        if not results:
            return "未找到相关信息"
        return "\n".join([result.get("content", "") for result in results])
    except Exception as e:
        return f"搜索失败: {str(e)}"


research_agent = create_agent(
    model=model,
    tools=[search_web],
    system_prompt="""你是一个勤奋的研究者，根据用户的主题进行深入研究。
    你会使用搜索工具在网络上查找相关信息，并整理成要点列表。""",
)

write_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""你是一个专业的写手，根据研究者提供的研究笔记撰写文章。
    你会将研究笔记整理成一篇结构清晰、内容丰富的文章。""",
)

review_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""你是一个严格的评审，根据写手提供的文章进行评审。
    你会指出文章中的不足之处，并提出改进建议。""",
)


def research_node(state: ResearchState) -> ResearchState:
    """研究节点，负责根据主题进行研究并整理研究笔记"""
    print("研究节点".center(50, "="))
    response = research_agent.invoke(
        {"messages": [HumanMessage(content=state["topic"])]}
    )
    notes = response["messages"][-1].content
    return {"research_notes": notes, "next_step": "write"}


def write_node(state: ResearchState) -> ResearchState:
    """写作节点，负责根据研究笔记撰写文章"""
    print("写作节点".center(50, "="))
    if state["review_feedback"]:
        feedback = state["review_feedback"]
        content = f'根据以下研究笔记撰写文章，并根据评审反馈进行改进，评审反馈如下:\n{feedback}\n研究笔记如下:\n{state["research_notes"]}'
    else:
        content = f'根据以下研究笔记撰写文章:\n{state["research_notes"]}'
    response = write_agent.invoke({"messages": [HumanMessage(content=content)]})
    draft = response["messages"][-1].content
    return {"draft": draft, "next_step": "review"}


def review_node(state: ResearchState) -> ResearchState:
    """评审节点，负责根据草稿进行评审并提供反馈"""
    print("评审节点".center(50, "="))
    response = review_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f'请评审以下文章草稿，并提供改进建议，如果有严重问题则明确标识出严重问题，如果没有严重问题，则明确标识**无严重问题**这几个字:\n{state["draft"]}'
                )
            ]
        }
    )
    # TODO: 正常这里应该接入 human-in-loop
    feedback = response["messages"][-1].content
    next_step = "finalize" if "无严重问题" in feedback else "write"
    print(f"next_step: {next_step}")
    print(f"feedback: \n{feedback}")
    if next_step == "finalize":
        return {
            "review_feedback": feedback,
            "next_step": next_step,
            "final_article": state["draft"],
        }
    return {"review_feedback": feedback, "next_step": next_step}


def coordinate_node(state: ResearchState) -> ResearchState:
    """协调节点，负责根据当前状态协调各个节点的执行"""
    print("协调节点".center(50, "="))
    next_step = state.get("next_step", "research")
    return {"next_step": next_step}


builder = StateGraph(ResearchState)
builder.add_node("research", research_node)
builder.add_node("write", write_node)
builder.add_node("review", review_node)
builder.add_node("coordinate", coordinate_node)

builder.set_entry_point("coordinate")
builder.add_conditional_edges(
    "coordinate",
    lambda state: state["next_step"],
    {"research": "research", "write": "write", "review": "review", "finalize": END},
)
builder.add_edge("research", "coordinate")
builder.add_edge("write", "coordinate")
builder.add_edge("review", "coordinate")

checkpointer = InMemorySaver()
agent = builder.compile(checkpointer=checkpointer)


def search_writing_team():

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "research_project_123",
        }
    }
    initial_state: ResearchState = {
        "topic": "人工智能在医疗领域的应用",
        "research_notes": "",
        "draft": "",
        "review_feedback": "",
        "final_article": "",
        "next_step": "research",
    }

    resp = agent.invoke(
        input=initial_state,
        config=config,
    )
    print("主题".center(50, "-"))
    print(initial_state["topic"])
    print("研究笔记".center(50, "-"))
    print(resp["research_notes"])
    print("文章草稿".center(50, "-"))
    print(resp["draft"])
    print("评审反馈".center(50, "-"))
    print(resp["review_feedback"])
    print("最终文章内容".center(50, "-"))
    print(resp["final_article"])


# uv run -m src.langgraph_demo.langgraph_multi_agent
if __name__ == "__main__":
    search_writing_team()
