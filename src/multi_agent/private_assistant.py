import os
from pathlib import Path
from pydoc import doc
from typing import List

from dotenv import load_dotenv

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


####################  工具 ####################
from langchain.tools import tool


@tool
def create_calendar_event(
    title: str, start_time: str, end_time: str, attendees: List[str], location: str
):
    """
    创建日历事件的工具函数。

    Args:
    - title: 事件标题
    - start_time: 事件开始时间，格式为 "YYYY-MM-DD HH:MM"
    - end_time: 事件结束时间，格式为 "YYYY-MM-DD HH:MM"
    - attendees: 参与者列表，每个参与者是一个字符串（如邮箱地址）
    - location: 事件地点

    Returns:
    - str: 创建成功的消息或错误信息
    """
    # 这里可以集成实际的日历API（如Google Calendar API）来创建事件
    # 目前仅返回一个模拟的成功消息
    return f"已创建事件 '{title}'，时间从 {start_time} 到 {end_time}，地点在 {location}，参与者包括 {", ".join(attendees)}。"


@tool
def send_email(to: List[str], subject: str, body: str, cc: List[str] = []):
    """
    发送电子邮件的工具函数。

    Args:
    - to: 收件人列表，每个收件人是一个字符串（如邮箱地址）
    - subject: 邮件主题
    - body: 邮件正文
    - cc: 抄送列表，每个抄送人是一个字符串（如邮箱地址）

    Returns:
    - str: 发送成功的消息或错误信息
    """
    # 这里可以集成实际的邮件发送API（如SMTP服务器或第三方邮件服务）来发送邮件
    # 目前仅返回一个模拟的成功消息
    print(
        f"发送邮件给 {', '.join(to)}，主题：{subject}，内容：{body}，抄送：{', '.join(cc)}"
    )
    return f"已向 {", ".join(to)} 发送主题为 '{subject}' 的邮件，抄送给 {", ".join(cc)}。邮件内容：\n{body}"


@tool
def get_available_time_slots(
    attendees: List[str], date: str, duration_minutes: int
) -> List[str]:
    """
    获取参与者在特定日期的可用时间段。

    Args:
    - attendees: 参与者列表，每个参与者是一个字符串（如邮箱地址）
    - date: 日期，格式为 "YYYY-MM-DD"
    - duration_minutes: 会议持续时间，单位为分钟

    Returns:
    - List[str]: 可用时间段列表，每个时间段是一个字符串（如 "HH:MM-HH:MM"）
    """
    # 这里可以集成实际的日历API来查询参与者的日程安排并计算可用时间段
    # 目前仅返回一些模拟的时间段
    return ["09:00-10:00", "11:00-12:00", "14:00-15:00", "16:00-17:00"]


def create_calendar_agent(model):
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    CALENDAR_AGENT_PROMPT = (
        "You are a calendar scheduling assistant. "
        "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
        "into proper ISO datetime formats. "
        "Use get_available_time_slots to check availability when needed. "
        "If there is no suitable time slot, stop and confirm unavailability in your response. "
        "Use create_calendar_event to schedule events. "
        "Always confirm what was scheduled in your final response."
    )

    calendar_agent = create_agent(
        model,
        tools=[create_calendar_event, get_available_time_slots],
        system_prompt=CALENDAR_AGENT_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "create_calendar_event": True,  # 在调用 create_calendar_event 工具前暂停，等待用户确认
                },
                description_prefix="即将创建以下日历事件，请确认内容是否正确：\n",
            )
        ],
    )
    return calendar_agent


def create_email_agent(model):
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    EMAIL_AGENT_PROMPT = (
        "You are an email assistant. "
        "Compose professional emails based on natural language requests. "
        "Extract recipient information and craft appropriate subject lines and body text. "
        "Use send_email to send the message. "
        "Always confirm what was sent in your final response."
    )

    email_agent = create_agent(
        model,
        tools=[send_email],
        system_prompt=EMAIL_AGENT_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": True,  # 在调用 send_email 工具前暂停，等待用户确认
                },
                description_prefix="即将发送以下邮件，请确认内容是否正确：\n",
            )
        ],
    )
    return email_agent


from langchain.tools import ToolRuntime
from langgraph.graph.state import CompiledStateGraph
from typing import TypedDict


class Context(TypedDict):
    calendar_agent: CompiledStateGraph
    email_agent: CompiledStateGraph


@tool
def schedule_event(request: str, runtime: ToolRuntime[Context]) -> str:
    """
    解析自然语言请求并安排会议。

    当用户想要创建，修改或检查日程安排。
    Args:
    - request: 包含会议安排信息的自然语言字符串，例如 "请帮我安排一个会议，主题是项目讨论，时间是下周二下午2点，参与者有Alice和Bob，地点在会议室A。"

    Returns:
    - str: 安排结果的消息
    """
    calendar_agent = runtime.context["calendar_agent"]
    result = calendar_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


@tool
def manage_email(request: str, runtime: ToolRuntime[Context]) -> str:
    """
    解析自然语言请求并管理邮件。

    当用户想要发送通知，提醒或者任何其他形式的邮件。
    Args:
    - request: 包含邮件信息的自然语言字符串，例如 "请帮我写一封邮件给Alice，主题是项目更新，内容是我们已经完成了第一阶段的工作，正在进入第二阶段。抄送给Bob。"

    Returns:
    - str: 邮件管理结果的消息
    """
    email_agent = runtime.context["email_agent"]
    result = email_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


def create_supervisor_agent(model):
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    MAIN_AGENT_PROMPT = (
        "You are a helpful personal assistant. "
        "You can schedule calendar events and send emails. "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "When a request involves multiple actions, use multiple tools in sequence."
    )

    supervisor_agent = create_agent(
        model,
        tools=[schedule_event, manage_email],
        system_prompt=MAIN_AGENT_PROMPT,
        context_schema=Context,
        checkpointer=InMemorySaver(),  # 使用内存保存工具调用历史记录
    )
    return supervisor_agent


def main():
    model = get_chat_model()
    email_agent = create_email_agent(model)
    calendar_agent = create_calendar_agent(model)
    supervisor_agent = create_supervisor_agent(model)

    # query = "请帮我安排一个会议，主题是项目讨论，时间是下周二下午2点，参与者有Alice和Bob，地点在会议室A。"
    # for step in calendar_agent.stream(
    #     {"messages": [{"role": "user", "content": query}]}
    # ):
    #     for update in step.values():
    #         for message in update.get("messages", []):
    #             message.pretty_print()

    # query = "请帮我写一封邮件给Alice，主题是项目更新，内容是我们已经完成了第一阶段的工作，正在进入第二阶段。抄送给Bob。"
    # for step in email_agent.stream({"messages": [{"role": "user", "content": query}]}):
    #     for update in step.values():
    #         for message in update.get("messages", []):
    #             message.pretty_print()

    # query = "请帮我安排一个会议，主题是项目讨论，时间是下周二下午2点，参与者有Alice和Bob，地点在会议室A。同时，请帮我写一封邮件通知他们会议的安排。"
    query = "下周二下午2点安排一个设计部门参加的为期1小时的会议，讨论新产品设计方案。会议地点在会议室A。请同时发一封邮件通知设计部门的同事，告诉他们会议的时间、地点和议题。"
    config = {"configurable": {"thread_id": "6"}}
    from langgraph.types import Interrupt

    interrupts: List[Interrupt] = []
    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        context={
            "calendar_agent": calendar_agent,
            "email_agent": email_agent,
        },
    ):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            else:
                interrupt_ = update[0]
                interrupts.append(interrupt_)
                print(f"\nINTERRUPTED: {interrupt_.id}")

    # 输出所有中断的详细信息
    # resumes = []
    resume = {}
    for interrupt_ in interrupts:
        print("中断消息确认".center(50, "-"))
        for request, review_config in zip(
            interrupt_.value["action_requests"], interrupt_.value["review_configs"]
        ):
            print(f"INTERRUPTED: {interrupt_.id}")
            print(f"{request["description"]}\n")
            if review_config["action_name"] == "send_email":
                ## TODO: 这里应该是前端交互之后传递回来的结果。目前先模拟一个用户确认的结果。
                ## 在interrupt的时候应该给前端足够的信息，主要是interrupt的信息，config里面的thread_id可以是用户的session_id或者用户id，不需要额外传递。
                ## 应该两个 interrupt 分开发送给前端。
                # resume = {}
                resume[interrupt_.id] = {
                    "decisions": [
                        # TODO: 父子 agent 找不到对应的 tool, 唯一的方式是 中断前置
                        # {
                        #     "type": "edit",
                        #     "edited_action": {
                        #         "name": "send_email",
                        #         "args": {
                        #             "to": [
                        #                 "design-team@google.com",
                        #                 "alice@google.com",
                        #             ],
                        #             "subject": "技术方案研讨会-会议通知",
                        #         },
                        #     },
                        # }
                        {
                            "type": "approve",
                        }
                    ]
                }
                # resumes.append(resume)
            elif review_config["action_name"] == "create_calendar_event":
                ## TODO: 这里应该是前端交互之后传递回来的结果。目前先模拟一个用户确认的结果。
                # resume = {}
                resume[interrupt_.id] = {
                    "decisions": [
                        {
                            "type": "approve",
                        }
                    ]
                }
                # resumes.append(resume)

    # 继续执行被中断的工具调用
    from langgraph.types import Command
    import time

    print("\n等待用户确认中...".center(50, "-"))
    # time.sleep(5)  # 模拟用户确认的等待时间
    # for resume in resumes:
    for step in supervisor_agent.stream(
        Command(resume=resume),
        config,
        context={
            "calendar_agent": calendar_agent,
            "email_agent": email_agent,
        },
    ):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    message.pretty_print()
            else:
                interrupt_ = update[0]
                interrupts.append(interrupt_)
                print(f"\nINTERRUPTED: {interrupt_.id}")


# uv run -m src.multi_agent.private_assistant
if __name__ == "__main__":
    main()
