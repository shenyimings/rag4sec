from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from embedding import create_embeddings
from colorama import Fore, Style, Back

import pdf4llm
import fitz


def pdf2md(pdf_path) -> str:
    pdf_doc = fitz.open(pdf_path)
    return pdf4llm.to_markdown(pdf_doc)


def split_doc(doc: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=64)
    return splitter.split_documents(doc)


def split_md(
    md_text: str, splitter_string=[("#", "H1"), ("##", "H2"), ("###", "H3")]
) -> Document:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=splitter_string, strip_headers=False
    )
    return splitter.split_text(md_text)


def add_documents(db: Chroma, filepath: str) -> bool:
    """Add documents to the database."""
    # 判断filepath后缀名是pdf还是md，都不是就return
    if filepath.endswith(".pdf"):
        md_text = pdf2md(filepath)
    elif filepath.endswith(".md"):
        with open(filepath, "r", encoding="utf8") as f:
            md_text = f.read()
    else:
        return False

    docs = split_doc(split_md(md_text=md_text))
    # embedding = create_embeddings()
    # , embedding=embedding, persist_directory="./vector_db"

    if db.add_documents(documents=docs):
        print(Fore.YELLOW + "Documents added successfully." + Style.RESET_ALL)
        # print(db.max_marginal_relevance_search(query="UDF提权", k=1))
        return True
    else:
        return False


# add_documents("./docs/BT面经.md")
