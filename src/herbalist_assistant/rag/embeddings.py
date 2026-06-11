from langchain_huggingface import HuggingFaceEmbeddings


def _best_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def create_embeddings(model_name: str):
    device = _best_device()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        # normalize_embeddings=True improves cosine similarity search quality,
        # especially for multilingual models.
        encode_kwargs={"normalize_embeddings": True},
    )