from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="./bge-large-zh-v1.5",
        model_kwargs={"device": "cuda"},
        # show_progress=True,
    )
