import os

from langchain_groq import ChatGroq


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please add it to your .env file or environment."
        )
    return api_key


def create_groq_llm(*, api_key: str, model_name: str, temperature: float):
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
    )

