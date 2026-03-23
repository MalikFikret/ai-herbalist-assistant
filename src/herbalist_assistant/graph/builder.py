from langgraph.graph import END, StateGraph

from herbalist_assistant.types import HerbalistState

from .nodes import make_generation_node, make_retrieval_node


def build_graph(*, retriever, llm, prompt_fn):
    retrieval_node = make_retrieval_node(retriever)
    generation_node = make_generation_node(llm, prompt_fn)

    graph = StateGraph(HerbalistState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generation", generation_node)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", END)

    return graph.compile()

