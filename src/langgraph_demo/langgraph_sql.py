import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import MessagesState, START

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
db_path = data_dir / "Chinook.db"


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


def init_db():
    import requests

    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

    if db_path.exists():
        print(f"{db_path} already exists, skipping download.")
    else:
        response = requests.get(url)
        if response.status_code == 200:
            db_path.write_bytes(response.content)
            print(f"数据库下载成功并保存到 {db_path}")
        else:
            print(f"数据库下载失败，状态码: {response.status_code}")


def get_db():
    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_uri(f"sqlite:///{db_path.as_posix()}")

    print(f"Dialect: {db.dialect}")
    print(f"Available tables: {db.get_usable_table_names()}")
    print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')
    return db


def create_sql_agent_tools(model, db):
    from langchain_community.agent_toolkits import SQLDatabaseToolkit

    toolkit = SQLDatabaseToolkit(db=db, llm=model)

    tools = toolkit.get_tools()

    for tool in tools:
        print(f"{tool.name}: {tool.description}\n")

    return tools


from langgraph.prebuilt import ToolNode

# 初始化数据库
init_db()

model = get_chat_model()
db = get_db()
tools = create_sql_agent_tools(model=model, db=db)

get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")


generate_query_system_prompt = """
您是一个专为与 SQL 数据库交互而设计的智能体。
给定一个输入问题，请创建一个语法正确的 {dialect} 查询语句来执行，
然后查看查询结果并返回答案。除非用户明确指定希望获取的具体示例数量，
否则始终将查询结果限制在最多 {top_k} 条。

您可以按相关列对结果进行排序，以返回数据库中最有意义的示例。
切勿查询特定表的所有列，只根据问题要求获取相关的列。

绝对不要对数据库执行任何 DML 语句（如 INSERT、UPDATE、DELETE、DROP 等）。
""".format(
    dialect=db.dialect,
    top_k=5,
)


def generate_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    # We do not force a tool call here, to allow the model to
    # respond naturally when it obtains the solution.
    llm_with_tools = model.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


check_query_system_prompt = """
您是一名注重细节的 SQL 专家。
请仔细检查 {dialect} 查询语句中是否存在常见错误，包括：
- 在 NULL 值上使用 NOT IN
- 应使用 UNION ALL 时误用 UNION
- 为不包含边界值的范围使用 BETWEEN
- 谓词中的数据类型不匹配
- 标识符引号使用不当
- 函数参数数量不正确
- 数据类型转换错误
- 连接查询中使用了错误的列

如果存在上述任何错误，请重写查询语句。如果无误，则直接返回原查询语句。

完成此检查后，您将调用适当的工具来执行查询。
""".format(dialect=db.dialect)


def check_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }

    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    # 处理不同的工具调用参数结构
    query_content = ""
    if "query" in tool_call["args"]:
        query_content = tool_call["args"]["query"]
    elif "sql_query" in tool_call["args"]:
        query_content = tool_call["args"]["sql_query"]
    else:
        # 如果找不到查询内容，使用工具调用的所有参数作为内容
        query_content = str(tool_call["args"])

    user_message = {"role": "user", "content": query_content}
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}


from langgraph.graph import END, StateGraph


def list_tables(state: MessagesState):
    """列出数据库中的所有表"""
    tables = db.get_usable_table_names()
    table_info = f"数据库中的表: {', '.join(tables)}"
    return {"messages": [{"role": "assistant", "content": table_info}]}


def call_get_schema(state: MessagesState):
    """调用获取数据库模式的工具"""
    from langchain_core.messages import AIMessage

    # 创建一个带有工具调用的消息
    message = AIMessage(
        content="正在获取数据库模式信息...",
        tool_calls=[
            {
                "name": "sql_db_schema",
                "args": {"table_names": "Track, Genre"},
                "id": "schema_call",
            }
        ],
    )
    return {"messages": [message]}


def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"


builder = StateGraph(MessagesState)
builder.add_node("list_tables", list_tables)
builder.add_node("call_get_schema", call_get_schema)
builder.add_node("get_schema", get_schema_node)
builder.add_node("generate_query", generate_query)
builder.add_node("check_query", check_query)
builder.add_node("run_query", run_query_node)

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

agent = builder.compile()


def main():
    question = "哪个流派的平均曲目长度最长？"

    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


# uv run -m src.langgraph_demo.langgraph_sql
if __name__ == "__main__":
    main()
