import operator
import os
from pathlib import Path
from pydoc import doc
from typing import Annotated, List, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime

# 准备环境变量
load_dotenv()
os.environ["LANGSMITH_PROJECT"] = __file__.split(sep="\\")[-1].split(".")[0]
embedding_model = os.getenv("EMBEDDING_MODEL")
hf_home = os.getenv("HF_HOME")
model_cache_dir = hf_home + "\\hub"

# 模型相关
api_key = os.getenv("DASHSCOPE_API_KEY", "")
api_base = os.getenv("DASHSCOPE_API_BASE", "")
model_name = os.getenv("DASHSCOPE_MODEL_NAME", "")

# 路径相关
root_dir = Path(__file__).parents[2]
data_dir = root_dir / "data"


def get_chat_model():
    from langchain_qwq import ChatQwen

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
    return model


class AgentInput(TypedDict):
    """
    子agent的输入state
    """

    query: str


class AgentOutput(TypedDict):
    """
    子agent的输出
    """

    source: str
    result: str


class Classification(TypedDict):
    """
    路由，要执行哪个agent，以及相关的query
    """

    source: Literal["github", "notion", "slack"]
    query: str


class RouterState(TypedDict):
    query: str
    classifications: List[Classification]
    results: Annotated[List[AgentOutput], operator.add]
    final_answer: str


@tool
def search_code(query: str, repo: str = "main") -> str:
    """
    搜索github仓库的代码
    """
    return f"在仓库 {repo} 中 找到了匹配的代码 '{query}': authentication middleware in src/auth.py"


@tool
def search_issues(query: str) -> str:
    """
    搜索github的issues和pull requests
    """
    return f"找到了匹配 '{query}' 的 issues (3条):  #142 (API auth docs), #89 (OAuth flow), #203 (token refresh)"


@tool
def search_prs(query: str) -> str:
    """
    搜索pull requests中的实现细节
    """
    return f"PR #156 添加了 JWT 验证, PR #178 更新了 OAuth scopes"


@tool
def search_notion(query: str) -> str:
    """
    搜索notion工作空间的文档
    """
    return f"找到文档: 'API Authentication Guide' - covers OAuth2 flow, API keys, and JWT tokens"


@tool
def get_page(page_id: str) -> str:
    """
    根据page_id获取对应页的内容
    """
    return f"页内容: Step-by-step authentication setup instructions"


@tool
def search_slack(query: str) -> str:
    """
    搜索 Slack 消息和主题讨论。
    """
    return f"在#engineering中找到相关讨论: 'Use Bearer tokens for API auth, see docs for refresh flow'"


@tool
def get_thread(thread_id: str) -> str:
    """获取特定的主题讨论内容"""
    return f"Thread discusses best practices for API key rotation"


from langchain.agents import create_agent


def get_github_agent(model):

    github_agent = create_agent(
        model,
        tools=[search_code, search_issues, search_prs],
        system_prompt=(
            "你是一名 GitHub 专家。通过搜索代码仓库、议题和拉取请求，"
            "回答关于代码、API 参考和实现细节的问题。"
        ),
    )
    return github_agent


def get_notion_agent(model):

    notion_agent = create_agent(
        model,
        tools=[search_notion, get_page],
        system_prompt=(
            "你是一名 Notion 专家。通过搜索组织的 Notion 工作空间，"
            "回答关于内部流程、政策及团队文档的问题。"
        ),
    )
    return notion_agent


def get_slack_agent(model):

    slack_agent = create_agent(
        model,
        tools=[search_slack, get_thread],
        system_prompt=(
            "你是一名 Slack 专家。通过搜索团队成员分享知识与解决方案的"
            "相关主题讨论和对话记录来回答问题。"
        ),
    )
    return slack_agent


from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """将用户查询分类为面向特定智能体的子问题的结果"""

    classifications: list[Classification] = Field(
        description="需要调用的智能体列表，包含其对应的目标子问题"
    )


######### langgraph 的 节点 #########

from langgraph.types import Send
from langgraph.runtime import Runtime
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import BaseChatModel


class Context:
    github_agent: CompiledStateGraph
    notion_agent: CompiledStateGraph
    slack_agent: CompiledStateGraph
    router_llm: BaseChatModel


def classify_query(state: RouterState, runtime: Runtime[Context]) -> dict:
    """对查询进行分类，并决定调用哪些智能体。"""
    router_llm = runtime.context["router_llm"]
    structured_llm = router_llm.with_structured_output(ClassificationResult)

    result = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": """请分析此查询，并确定需要咨询哪些知识库。
为每个相关来源生成针对该来源优化的目标子问题。

可用来源：
- github: 代码、API 参考、实现细节、议题、拉取请求
- notion: 内部文档、流程、政策、团队维基
- slack: 团队讨论、非正式知识分享、近期对话记录

仅返回与查询相关的来源。每个来源都应包含针对该特定知识领域优化的目标子问题。

例如，对于"如何进行 API 请求鉴权？"：
- github: "存在哪些鉴权相关代码？请搜索鉴权中间件、JWT 处理逻辑"
- notion: "存在哪些鉴权相关文档？请查找 API 鉴权指南"
（此例中未包含 slack，因为该技术问题不相关）""",
            },
            {"role": "user", "content": state["query"]},
        ]
    )

    return {"classifications": result.classifications}


# 条件边
def route_to_agents(state: RouterState) -> list[Send]:
    """根据classifications并发执行节点"""
    return [Send(c["source"], {"query": c["query"]}) for c in state["classifications"]]


def query_github(state: AgentInput, runtime: Runtime[Context]) -> dict:
    """Query the GitHub agent."""
    github_agent = runtime.context["github_agent"]
    result = github_agent.invoke(
        {"messages": [{"role": "user", "content": state["query"]}]}
    )
    return {"results": [{"source": "github", "result": result["messages"][-1].content}]}


def query_notion(state: AgentInput, runtime: Runtime[Context]) -> dict:
    """Query the Notion agent."""
    notion_agent = runtime.context["notion_agent"]
    result = notion_agent.invoke(
        {"messages": [{"role": "user", "content": state["query"]}]}
    )
    return {"results": [{"source": "notion", "result": result["messages"][-1].content}]}


def query_slack(state: AgentInput, runtime: Runtime[Context]) -> dict:
    """Query the Slack agent."""
    slack_agent = runtime.context["slack_agent"]
    result = slack_agent.invoke(
        {"messages": [{"role": "user", "content": state["query"]}]}
    )
    return {"results": [{"source": "slack", "result": result["messages"][-1].content}]}


def synthesize_results(state: RouterState, runtime: Runtime[Context]) -> dict:
    """将所有智能体的结果整合成连贯的回答。"""
    if not state["results"]:
        return {"final_answer": "未从任何知识源中找到相关结果。"}

    # 格式化结果以进行整合
    formatted = [
        f"**来自 {r['source'].title()}：**\n{r['result']}" for r in state["results"]
    ]
    router_llm = runtime.context.router_llm
    synthesis_response = router_llm.invoke(
        [
            {
                "role": "system",
                "content": f"""请整合以下搜索结果，以回答原始问题："{state['query']}"

- 汇总多来源信息，避免冗余
- 突出最相关且可操作的信息
- 指出不同来源间的差异
- 保持回答简洁、结构清晰""",
            },
            {"role": "user", "content": "\n\n".join(formatted)},
        ]
    )

    return {"final_answer": synthesis_response.content}


from langgraph.graph import StateGraph, START, END


def create_workflow():

    workflow = (
        StateGraph(RouterState)
        .add_node("classify", classify_query)
        .add_node("github", query_github)
        .add_node("notion", query_notion)
        .add_node("slack", query_slack)
        .add_node("synthesize", synthesize_results)
        .add_edge(START, "classify")
        .add_conditional_edges(
            "classify", route_to_agents, ["github", "notion", "slack"]
        )
        .add_edge("github", "synthesize")
        .add_edge("notion", "synthesize")
        .add_edge("slack", "synthesize")
        .add_edge("synthesize", END)
        .compile()
    )
    return workflow


def main():
    workflow = create_workflow()
    router_llm = get_chat_model()
    github_agent = get_github_agent(model=router_llm)
    notion_agent = get_notion_agent(model=router_llm)
    slack_agent = get_slack_agent(model=router_llm)
    result = workflow.invoke(
        {"query": "How do I authenticate API requests?"},
        context={
            "github_agent": github_agent,
            "notion_agent": notion_agent,
            "slack_agent": slack_agent,
            "router_llm": router_llm,
        },
    )

    print("Original query:", result["query"])
    print("\nClassifications:")
    for c in result["classifications"]:
        print(f"  {c['source']}: {c['query']}")
    print("\n" + "=" * 60 + "\n")
    print("Final Answer:")
    print(result["final_answer"])


# uv run -m src.multi_agent.router_rag
if __name__ == "__main__":
    # TODO: 这个有bug
    # main()
    pass
