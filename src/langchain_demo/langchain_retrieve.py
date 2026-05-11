import os
from pathlib import Path
from pydoc import doc

from dotenv import load_dotenv

# 准备环境变量
load_dotenv()
os.environ["LANGSMITH_PROJECT"] = __file__.split(sep="\\")[-1].split(".")[0]
embedding_model = os.getenv("EMBEDDING_MODEL")
hf_home = os.getenv("HF_HOME")
model_cache_dir = hf_home + "\\hub"

# 路径相关
root_dir = Path(__file__).parents[2]
data_dir = root_dir / "data"


def load_documents():
    from langchain_community.document_loaders import PyPDFLoader

    file_path = data_dir / "nke-10k-2023.pdf"

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"成功加载文档，文档数量: {len(documents)}")
    return documents


def split_documents(documents):
    # 更好的方式是使用 langchain_classic 里面的 ParentDocumentRetriever
    # from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    split_texts = text_splitter.split_documents(documents)
    print(f"成功分割文档，分割后的文本数量: {len(split_texts)}")
    return split_texts


def get_embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    # 归一化嵌入向量
    # 归一化后，嵌入向量的长度为1，这可以提高相似度计算的准确性
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        # show_progress=True,
        cache_folder=model_cache_dir,  # 缓存模型文件, 避免重复下载
    )


def create_vectorstore(split_texts, embedding_model):
    from langchain_community.vectorstores import InMemoryVectorStore

    vector_store = InMemoryVectorStore.from_documents(
        split_texts, embedding_model, show_progress=True
    )
    return vector_store


def main():

    # 1. 加载文档
    print("加载文档".center(60, "-"))
    documents = load_documents()
    for document in documents[:2]:
        print(document.page_content[:100])
        print(document.metadata, end="\n\n")

    # 2. 分割文档
    print("分割文档".center(60, "-"))
    split_texts = split_documents(documents)
    for text in split_texts[:2]:
        print(text.page_content[:100])
        print(text.metadata, end="\n\n")

    # 3. 获取嵌入模型
    print("获取嵌入模型".center(60, "-"))
    embedding_model = get_embedding_model()

    # 4. 创建vectorstore
    print("创建vectorstore".center(60, "-"))
    vector_store = create_vectorstore(split_texts, embedding_model)

    # results = vector_store.similarity_search(
    #     "How many distribution centers does Nike have in the US?"
    # )
    # results = vector_store.similarity_search_with_score(
    #     "How many distribution centers does Nike have in the US?"
    # )
    # print("查询结果".center(60, "-"))
    # for result in results:
    #     print(result[0].page_content[:100])
    #     print(result[0].metadata)
    #     print("相似度得分:", result[1], end="\n\n")

    # 5. 创建retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    )
    results = retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ]
    )
    print("查询结果".center(60, "-"))
    for result in results:
        print(result[0].page_content[:100])
        print(result[0].metadata, end="\n\n")


# uv run -m src.langchain_demo.langchain_retrieve
if __name__ == "__main__":
    main()
