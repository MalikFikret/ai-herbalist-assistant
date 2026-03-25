from typing import Callable
from pathlib import Path

from herbalist_assistant.types import HerbalistState


def make_retrieval_node(retriever) -> Callable[[HerbalistState], HerbalistState]:
    def retrieval_node(state: HerbalistState) -> HerbalistState:
        question = state.get("question", "")
        if not question:
            return state

        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs) if docs else ""
        sources = []
        for doc in docs or []:
            source = doc.metadata.get("source", "")
            if source:
                sources.append(Path(str(source)).name)
        # Preserve order while removing duplicates.
        sources = list(dict.fromkeys(sources))
        return {
            "question": question,
            "context": context,
            "answer": state.get("answer", ""),
            "sources": sources,
        }

    return retrieval_node


def make_generation_node(llm, prompt_fn) -> Callable[[HerbalistState], HerbalistState]:
    def generation_node(state: HerbalistState) -> HerbalistState:
        question = state.get("question", "")
        context = state.get("context", "")
        prompt = prompt_fn(question=question, context=context)

        response = llm.invoke(prompt)
        answer = getattr(response, "content", str(response))
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "sources": state.get("sources", []),
        }

    return generation_node

