from typing import List, TypedDict


class HerbalistState(TypedDict):
    question: str
    context: str
    answer: str
    sources: List[str]

