from config import VECTOR_DB_DIR
from rag.graph import build_graph

_graph_cache = {}


def _get_graph(store_name: str = VECTOR_DB_DIR):
    if store_name not in _graph_cache:
        _graph_cache[store_name] = build_graph(store_name)
    return _graph_cache[store_name]


def answer_question(question: str, store_name: str = VECTOR_DB_DIR) -> dict:
    """
    Run the RAG graph for a single question.

    @param question User's question about the Romanian fiscal code
    @return dict with "answer" (str) and "citations" (list of {text, metadata})
    """
    graph = _get_graph(store_name)
    result = graph.invoke({"question": question})
    return {"answer": result["answer"], "citations": result.get("citations", [])}
