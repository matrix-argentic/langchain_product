import os
from pathlib import Path
from pydoc import doc
from typing import List, Literal, NotRequired

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

# 工作流步骤
# warranty_collectior: 质检信息收集
# issue_classifier: 问题分类
# resolution_specialist: 问题解决方案
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

from langchain.agents import AgentState
from langgraph.types import Command
from langchain.messages import ToolMessage


class SupportState(AgentState):
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[
        Literal["in_warranty", "out_of_warranty"]
    ]  # 是否在质保内
    issue_type: NotRequired[Literal["hardware", "software"]]


@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """
    记录客户的质保状态，并转向问题分类
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
):
    """
    记录问题并且转向解决问题步骤
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool
def escalate_to_human(reason: str) -> str:
    """
    将该工单升级转给人工客服专员处理。
    """
    return f"Escalating to human support. Reason: {reason}"


@tool
def provide_solution(solution: str) -> str:
    """
    针对客户的问题提供解决方案
    """
    return f"Solution provided: {solution}"


@tool
def go_back_to_warranty() -> Command:
    """返回到质检信息收集步骤"""
    return Command(update={"current_step": "warranty_collector"})


@tool
def go_back_to_classification() -> Command:
    """返回到问题分类步骤"""
    return Command(update={"current_step": "issue_classifier"})


WARRANTY_COLLECTOR_PROMPT = """您是一名帮助处理设备问题的客户支持专员。

当前阶段：warranty_collector

在此步骤中，您需要：
1. 热情问候客户
2. 询问其设备是否在保修期内
3. 使用 record_warranty_status 记录客户答复并进入下一阶段

请保持对话自然、友好。避免一次性提出多个问题。"""

ISSUE_CLASSIFIER_PROMPT = """您是一名帮助处理设备问题的客户支持专员。

当前阶段：issue_classifier
客户信息：保修状态为 {warranty_status}

在此步骤中，您需要：
1. 请客户描述其遇到的问题
2. 判断是否为硬件问题（物理损坏、部件故障）或软件问题（应用崩溃、性能问题）
3. 使用 record_issue_type 记录问题分类并进入下一阶段

若情况不明确，请在分类前进一步询问以明确问题。"""

RESOLUTION_SPECIALIST_PROMPT = """您是一名帮助处理设备问题的客户支持专员。

当前阶段：resolution_specialist
客户信息：保修状态为{warranty_status}，问题类型为{issue_type}

在此步骤中，您需要：
1. 对于软件问题：通过 provide_solution 提供故障排查步骤
2. 对于硬件问题：
   - 若在保修期内：通过 provide_solution 说明保修维修流程
   - 若已过保修期：通过 escalate_to_human 转接人工服务咨询付费维修选项

如果客户指出任何信息有误，请使用：
- go_back_to_warranty 以修正保修状态
- go_back_to_classification 以修正问题类型

提供的解决方案应具体明确、切实有用。"""

# 不同步骤的提示词和工具
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [
            provide_solution,
            escalate_to_human,
            go_back_to_warranty,
            go_back_to_classification,
        ],
        "requires": ["warranty_status", "issue_type"],
    },
}


from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    # 获取当前步骤
    current_step = request.state.get("current_step", "warranty_collector")

    # 动态加载配置
    stage_config = STEP_CONFIG[current_step]

    # 验证必须的state是否存在
    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # 格式化系统提示词
    system_prompt = stage_config["prompt"].format(**request.state)

    # 重写提示词和工具
    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )

    return handler(request)


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


def get_customer_agent(model):
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain.agents.middleware import SummarizationMiddleware

    all_tools = [
        record_warranty_status,
        record_issue_type,
        provide_solution,
        escalate_to_human,
    ]

    # Create the agent with step-based configuration
    agent = create_agent(
        model,
        tools=all_tools,
        state_schema=SupportState,
        middleware=[
            apply_step_config,
            SummarizationMiddleware(
                model=model, trigger=("tokens", 4000), keep=("messages", 10)
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    return agent


# uv run -m src.multi_agent.customer_support
if __name__ == "__main__":
    from langchain_core.utils.uuid import uuid7
    from langchain.messages import HumanMessage

    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    model = get_chat_model()
    agent = get_customer_agent(model=model)

    print("=== 第一轮：质保收集 ===")
    result = agent.invoke({"messages": [HumanMessage("你好，我的iPhone坏了")]}, config)
    for msg in result["messages"]:
        msg.pretty_print()

    print("\n=== 第二轮：回复质保 ===")
    result = agent.invoke({"messages": [HumanMessage("是的，还在质保期内")]}, config)
    for msg in result["messages"]:
        msg.pretty_print()
    print(f"Current step: {result.get('current_step')}")

    print("\n=== 第三轮：用户描述问题 ===")
    result = agent.invoke(
        {"messages": [HumanMessage("屏幕掉地上摔坏了")]},
        config,
    )
    for msg in result["messages"]:
        msg.pretty_print()
    print(f"Current step: {result.get('current_step')}")

    print("\n=== 第四轮：解决方案 ===")
    result = agent.invoke({"messages": [HumanMessage("我需要做什么?")]}, config)
    for msg in result["messages"]:
        msg.pretty_print()
    print(f"Current step: {result.get('current_step')}")

    print("\n== 第五轮：用户提示记错信息 ===")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "Actually, I made a mistake - my device is out of warranty"
                )
            ]
        },
        config,
    )
    for msg in result["messages"]:
        msg.pretty_print()
    print(f"Current step: {result.get('current_step')}")
