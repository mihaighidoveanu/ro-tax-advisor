from typing import List, TypedDict


class Citation(TypedDict):
    text: str
    metadata: dict


class AgentState(TypedDict, total=False):
    question: str
    citations: List[Citation]
    answer: str
