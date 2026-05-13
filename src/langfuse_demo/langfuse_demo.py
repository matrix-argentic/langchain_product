from importlib import metadata
import os
from typing import Literal, TypedDict
from uuid import uuid4
from dotenv import load_dotenv
from langchain_qwq import ChatQwen
from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import Anonymize, PromptInjection, TokenLimit, Toxicity
from llm_guard.output_scanners import Deanonymize, NoRefusal, Relevance, Sensitive
from llm_guard.vault import Vault
from langchain_core.prompts import PromptTemplate

# 加载环境变量
load_dotenv()
# 不同环境
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "dev"


def langchain_demo():
    # 1. 初始化 langfuse
    langfuse_handler = CallbackHandler()
    langfuse = get_client()

    # 2. 模型相关配置
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    api_base = os.getenv("DASHSCOPE_API_BASE", "")
    model_name = os.getenv("DASHSCOPE_MODEL_NAME", "")

    # 3. 初始化模型
    model = ChatQwen(
        model=model_name,
        max_tokens=3000,
        timeout=None,
        max_retries=2,
        api_key=api_key,
        base_url=api_base,
        enable_thinking=False,  # 关闭深度思考
    )

    with langfuse.start_as_current_observation(
        as_type="span", name="langchain-cli"
    ) as langchain_cli_span:
        session_id = str(uuid4())
        input = "给我讲一个关于猫的笑话"
        langchain_cli_span.update(input=input)
        with propagate_attributes(
            session_id=session_id,
            user_id="user_123456",
        ) as user_span:
            # 4. 调用模型
            from langchain_core.messages import HumanMessage

            response = model.invoke(
                [HumanMessage(content=input)],
                config={
                    "callbacks": [langfuse_handler],
                    "configurable": {"thread_id": session_id},
                    "metadata": {"langfuse_tags": ["simple_call"]},  # tags
                },
            )
            langchain_cli_span.update(output=response)
            print(response.text)
            # 直接更新, 这几个更新的都是 langchain_cli_span的属性，最终以最后一个为主了
            langchain_cli_span.update(
                name="subagent",
                metadata={"staging": "parsing"},
                level="WARNING",  # 日志等级
            )
            langchain_cli_span.update(name="subagent", metadata={"staging": "retrieve"})
            langchain_cli_span.update(name="subagent", metadata={"staging": "answer"})

            trace_id = langfuse.get_current_trace_id()
            observation_id = langfuse.get_current_observation_id()
            print(f"trace_id: {trace_id}")
            print(f"observation_od: {observation_id}")

            # TODO: 这个trace_id, observation_id可能需要维护一下，后面通过这个来添加score
            # langfuse.create_score(
            #     name="ticket-joke",
            #     trace_id=trace_id,
            #     observation_id=observation_id,
            #     value=1,
            #     comment="这个笑话生成的很有趣，我喜欢",
            # )


class ChatState(TypedDict):
    query: str
    answer: str
    stage: Literal["rewrite", "orchester", "rag", "agent", "answer"]


def rewrite_node(state: ChatState):
    return {"stage": "rewrite"}


def orchester_node(state: ChatState):
    return {"stage": "orchester"}


def rag_node(state: ChatState):
    return {"stage": "rag"}


def agent_node(state: ChatState):
    return {"stage": "agent"}


def answer_node(state: ChatState):
    return {
        "stage": "answer",
        "answer": "明天天气晴，气温舒适，适合出游，要我为您推荐附近出游的路线么？",
    }


# 支持 图显示。。
def langgraph_demo():

    # 设置追踪的版本，可以全局设置，也可以在某个节点设置
    langfuse = Langfuse(release="v0.1.0")

    # 1. langfuse
    langfuse = get_client()
    langfuse_handler = CallbackHandler()

    # 2. graph
    builder = StateGraph(ChatState)
    graph = (
        builder.add_node(rewrite_node)
        .add_node(orchester_node)
        .add_node(rag_node)
        .add_node(agent_node)
        .add_node(answer_node)
        .add_edge(START, "rewrite_node")
        .add_edge("rewrite_node", "orchester_node")
        .add_edge("orchester_node", "rag_node")
        .add_edge("rag_node", "agent_node")
        .add_edge("agent_node", "answer_node")
        .add_edge("answer_node", END)
        .compile()
    )

    # 用户从接口提的问题
    query = "明天的天气不错"

    session_id = str(uuid4())
    user_id = "user_123"
    metadata = {
        "thread_id": session_id,
        "user": "zhangsan",
        "user_id": user_id,
        "level": 1,
    }

    input = {"query": query}

    model_name = os.getenv("DASHSCOPE_MODEL_NAME", "")

    with langfuse.start_as_current_observation(
        as_type="span", name="graph", level="DEFAULT", metadata=metadata, input=input
    ) as graph_span:
        with propagate_attributes(session_id=session_id, user_id=user_id):
            with langfuse.start_as_current_observation(
                as_type="generation", name="graph-genration", model=model_name
            ) as generation:
                config = {
                    "configuration": {"thread_id": session_id},
                    "callbacks": [langfuse_handler],
                }
                resp = graph.invoke(input=input, config=config)
                generation.update(output=resp)
                graph_span.update(output=resp)
                trace_id = langfuse.get_current_trace_id()
                observation_id = langfuse.get_current_observation_id()
                # TODO: 避免重复打分的关键就是score_id的幂等处理
                score_name = "graph"
                score_id = f"{trace_id}-{score_name}"
                print(f"trace_id: {trace_id}")
                print(f"observation_od: {observation_id}")
                print(f"score_id: {score_id}")

    # TODO: 在结束前保证不丢失
    # LANGFUSE_FLUSH_AT 当前需要批量处理的最大事件数量
    # LANGFUSE_FLUSH_INTERVAL 发送批次前的最大等待时间为几秒钟。
    # LANGFUSE_FLUSH_INTERVAL=1 LANGFUSE_FLUSH_AT=10 表示每满10条或者1秒就发送一次flush
    langfuse.flush()


def langchain_score_comment():

    # 避免重复打分,还支持更新
    langfuse = get_client()
    trace_id = "0a31a29773437a0c9c8ba20d62a5d49f"
    score_name = "ticket-joke-1"
    observation_id = "49b294fe50bb5121"
    # TODO: 避免重复打分的关键就是score_id的幂等处理
    score_id = f"{trace_id}-{score_name}"
    langfuse.create_score(
        score_id=score_id,
        name=score_name,
        trace_id=trace_id,
        observation_id=observation_id,
        value=1,
        comment="这个笑话我很喜欢",
        metadata={
            "name": "zhangsan",
            "user_id": 10,
            "level": 1,
        },
    )


def llm_guard_ner_demo():
    vault = Vault()
    scanner = Anonymize(vault)

    # input_scanners = [Anonymize(vault), Toxicity(), TokenLimit(), PromptInjection()]
    # output_scanners = [Deanonymize(vault), NoRefusal(), Relevance(), Sensitive()]
    # Langchain 的 Middleware 里面已经有这个了 PII 检测
    def mask_function(data, **kwargs):
        if isinstance(data, str):
            sanitized_data, is_valid, risk_score = scanner.scan(data)
            return sanitized_data
        return data

    langfuse = Langfuse(mask=mask_function)
    langfuse = get_client()

    # xxxx


def mcp_demo():
    # MCP通过其场约定支持上下文传播。通过将 OpenTelemetry 上下文（W3C Trace Context 格式）注入工具调用，您可以将客户端和服务器跟踪链接起来：_meta

    # 在客户端提取当前的跟踪上下文
    # 将其注入MCP工具调用的字段_meta
    # 在服务器端提取并恢复上下文
    # 所有服务器操作都继承客户端的跟踪上下文
    # https://github.com/langfuse/langfuse-examples/tree/main/applications/mcp-tracing/src

    # @mcp.tool()
    # @with_otel_context_from_meta
    # @observe(name="trace-from-mcp-server")
    # def search(query: str, _meta: dict = None) -> str:
    #     """Search for web pages using Exa"""

    #     response = exa.search_and_contents(
    #         query, type="auto", num_results=1, highlights=True
    #     )

    #     langfuse.update_current_trace(
    #         metadata={
    #             "num_results": len(response.results),
    #             "results": [{"title": r.title, "url": r.url} for r in response.results],
    #         }
    #     )

    #     result = "".join(
    #         [
    #             f"<Title id={idx}>{r.title}</Title>"
    #             f"<URL id={idx}>{r.url}</URL>"
    #             f"<Highlight id={idx}>{''.join(r.highlights)}</Highlight>"
    #             for idx, r in enumerate(response.results)
    #         ]
    #     )

    #     langfuse.flush()
    #     return result
    pass


def langfuse_prompt_demo():
    langfuse = get_client()
    label = "develop"
    print("=" * 30)
    try:
        prompt = langfuse.get_prompt("study_planner", label="latest")
    except Exception as e:
        print(f"prompt 加载失败: {e}")
        return
    print(prompt.version)
    args = {
        "outline": "Python 基础语法学习：变量、数据类型、控制流、函数、文件操作",
        "goal": "掌握 Python 基础，能够编写简单的脚本程序",
        "hours_per_day": 1,
        "total_days": 30,
        "learning_style": "自学为主，喜欢看文档和视频教程",
        "prior_knowledge": "无编程经验，计算机操作熟练",
        "preferred_resources": "在线教程 + 练习平台",
    }
    compiled_prompt = prompt.compile(**args)
    # langchain_prompt = prompt.get_langchain_prompt()

    # template = PromptTemplate.from_template(compiled_prompt)
    print(compiled_prompt)
    # TODO: 这个可以在后台设置model，temeperature 等影响模型输出的参数。
    print(prompt.config)


# uv run -m src.langfuse_demo.langfuse_demo
if __name__ == "__main__":
    # langchain_demo()
    # langchain_score_comment()
    langgraph_demo()
    # langfuse_prompt_demo()
