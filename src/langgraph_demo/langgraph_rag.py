import os
from pathlib import Path
from pydoc import doc
from typing import Annotated, List, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langgraph.graph import MessagesState

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


def load_documents():
    os.environ["USER_AGENT"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
    from langchain_community.document_loaders import WebBaseLoader

    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    return docs


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs: List[List[Document]]):
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


from langchain_core.vectorstores import InMemoryVectorStore
import torch
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(documents: List[Document]):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    # 归一化嵌入向量
    # 归一化后，嵌入向量的长度为1，这可以提高相似度计算的准确性
    encode_kwargs = {"normalize_embeddings": True}
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        # show_progress=True,
        cache_folder=model_cache_dir,  # 缓存模型文件, 避免重复下载
    )
    vector_store = InMemoryVectorStore.from_documents(
        documents=documents, embedding=embedding
    )
    vector_store.as_retriever()
    return vector_store


from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chat_models import BaseChatModel


class Context(TypedDict):
    retriever: VectorStoreRetriever
    model: BaseChatModel


@tool
def retrieve_blog_posts(query: str, runtime: ToolRuntime[Context]) -> str:
    """搜索并返回关于 Lilian Weng 博客文章的信息。"""
    retriever = runtime.context["retriever"]
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


######## 生成查询docs的结点 #########

from langgraph.runtime import Runtime


def generate_query_or_respond(state: MessagesState, runtime: Runtime[Context]):
    """
    调用模型基于当前状态生成响应。根据问题内容，它将决定使用检索工具进行检索，或直接向用户做出回复。
    """
    model = runtime.context["model"]
    response = model.bind_tools([retrieve_blog_posts]).invoke(state["messages"])
    return {"messages": [response]}


######## 评价打分的的结点 #########


GRADE_PROMPT = (
    "您是一名评估员，负责评估检索到的文档与用户问题的相关性。\n"
    "以下是检索到的文档：\n\n{context}\n\n"
    "以下是用户的问题：{question}\n"
    "如果文档包含与用户问题相关的关键词或语义含义，则将其评为相关。\n"
    "请给出二元评分 'yes' 或 'no' 以表示文档是否与问题相关。"
)


from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def grade_documents(
    state: MessagesState, runtime: Runtime[Context]
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    model = runtime.context["model"]
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


######## 重写问题的结点 #########

from langchain.messages import HumanMessage

REWRITE_PROMPT = (
    "请分析输入内容，尝试推理其背后的语义意图/含义。\n"
    "以下是初始问题："
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "请重构一个优化后的问题："
)


def rewrite_question(state: MessagesState, runtime: Runtime[Context]):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    model = runtime.context["model"]
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


######## 生成答案的结点 #########

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState, runtime: Runtime[Context]):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    model = runtime.context["model"]
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


######## 组装workflow #########
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_blog_posts]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()


def main():
    documents = load_documents()
    documents = split_documents(documents)
    vector_store = create_vector_store(documents=documents)
    retriever = vector_store.as_retriever()

    model = get_chat_model()
    from langchain_core.utils.uuid import uuid7

    for chunk in graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                }
            ]
        },
        context={"model": model, "retriever": retriever},
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")


# uv run -m src.langgraph_demo.langgraph_rag
if __name__ == "__main__":
    main()
