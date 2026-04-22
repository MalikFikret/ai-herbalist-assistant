from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

