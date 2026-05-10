import os
from pathlib import Path
from pydoc import doc

from dotenv import load_dotenv

# 准备环境变量
load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "langchain_sql"
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


def get_agent(tools, model, dialect):
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langgraph.checkpoint.memory import InMemorySaver

    system_prompt = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    database.

    To start you should ALWAYS look at the tables in the database to see what you
    can query. Do NOT skip this step.

    Then you should query the schema of the most relevant tables.
    """.format(
        dialect=dialect,
        top_k=5,
    )
    agent = create_agent(
        model,
        tools,
        system_prompt=system_prompt,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"sql_db_query": True},
                description_prefix="Tool execution pending approval",
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    return agent


def main():
    # 1. 初始化数据库
    print("初始化数据库".center(60, "-"))
    init_db()

    # 2. 获取数据库连接
    print("获取数据库连接".center(60, "-"))
    db = get_db()

    # 3. 获取聊天模型
    print("获取聊天模型".center(60, "-"))
    model = get_chat_model()

    # 4. 创建SQL agent tools
    print("创建SQL agent tools".center(60, "-"))
    tools = create_sql_agent_tools(model, db)

    # 5. 创建agent
    print("创建agent".center(60, "-"))
    agent = get_agent(tools, model, db.dialect)

    # 6. 测试agent
    question = "Which genre on average has the longest tracks?"
    config = {"configurable": {"thread_id": "1"}}
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="values",
    ):
        if "__interrupt__" in step:
            print("INTERRUPTED:")
            interrupt = step["__interrupt__"][0]
            for request in interrupt.value["action_requests"]:
                print(request["description"])
        elif "messages" in step:
            step["messages"][-1].pretty_print()
        else:
            pass

    # 7. 继续执行agent
    from langgraph.types import Command

    while True:
        input_str = input("输入 'approve' 来批准agent继续执行: ")
        if input_str.lower() == "approve":
            break
        else:
            print("无效输入，请输入 'approve' 来继续。")

    for step in agent.stream(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
        stream_mode="values",
    ):
        if "messages" in step:
            step["messages"][-1].pretty_print()
        if "__interrupt__" in step:
            print("INTERRUPTED:")
            interrupt = step["__interrupt__"][0]
            for request in interrupt.value["action_requests"]:
                print(request["description"])
        else:
            pass


# uv run -m src.langchain_demo.langchain_sql
if __name__ == "__main__":
    main()
