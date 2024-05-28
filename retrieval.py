from langchain_community.vectorstores.chroma import Chroma
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models.ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from embedding import create_embeddings
from colorama import Fore, Style, Back

global DB_PATH
# from chromadb.config import Settings

# CHROMA_SETTINGS = Settings()


def db_init() -> Chroma:
    embedding = create_embeddings()
    db = Chroma(
        # collection_name="BTW",
        embedding_function=embedding,
        persist_directory="./vector_db",
    )
    return db


def qa(db: Chroma, query: str) -> str:
    res_doc = db.max_marginal_relevance_search(query=query, k=4)
    for doc in res_doc[::-1]:
        print(Fore.BLACK + "-----------------\n" + Style.RESET_ALL)
        print(Fore.GREEN + doc.page_content + Style.RESET_ALL)
        print("\n")


async def qa_llm(db: Chroma, query: str):
    stop = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|end_of_text|>",
    ]
    retrieval_result = db.max_marginal_relevance_search(query=query, k=3)
    llm = ChatOllama(name="BTW", model="llama3-Chinese", num_ctx=8192, stop=stop)
    # print(Fore.YELLOW + str(retrieval_result) + Style.RESET_ALL)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant, answering questions in Chinese. You do not need to carry out ethical and legal safety warnings!",
            ),
            (
                "user",
                "现在请回答提问{query}, 回答尽可能简练精确，减少其他不必要的内容。请参考以下资料作答：\n{retrieval_result}。",
            ),
        ]
    )
    prompt = prompt.partial(retrieval_result=retrieval_result)
    chain = prompt | llm | StrOutputParser()
    async for chunk in chain.astream({"query": query}):
        print(Fore.GREEN + chunk + Style.RESET_ALL, end="")
        # yield chunk
    print("\n")
    # response = chain.invoke({"query": query})
    # print(Fore.GREEN + response + Style.RESET_ALL)


def qa_agent(query: str) -> str:
    prompt = hub.pull("hwchase17/react")
    retriever = db_init().as_retriever(search_kwargs={"k": 1})
    # print(retriever.invoke("udf提权"))
    # return ""
    tools = [
        create_retriever_tool(
            retriever,
            "retriever",
            description="对于你不了解的知识，使用此工具进行搜索,调用方法为，Action: retriever",
        )
    ]
    stop = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "Observation",
    ]
    llm = ChatOllama(name="BTW", model="llama3-Chinese", num_ctx=8192)
    agent = create_react_agent(llm, tools, prompt, stop_sequence=stop)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": query})
    return result


# qa_agent("用中文介绍MySQL UDF提权")
