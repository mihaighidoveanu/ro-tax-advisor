from langgraph.graph import END, START, StateGraph

from config import RETRIEVER_TOP_K, VECTOR_DB_DIR
from ingestion.vectorstore import load_vectorstore
from rag.llm import get_llm
from rag.state import AgentState

SYSTEM_PROMPT = (
    "Esti un asistent specializat in Codul fiscal romanesc. Raspunde la intrebarea "
    "utilizatorului folosind EXCLUSIV informatiile din contextul furnizat mai jos, citand "
    "articolul relevant (ex: 'conform Articolul 8'). Daca informatia nu se regaseste in "
    "context, spune ca nu poti raspunde pe baza documentelor disponibile."
)


def _format_citation_label(metadata: dict) -> str:
    parts = []
    if metadata.get("title"):
        parts.append(f"Titlul {metadata.get('title_no', '')} - {metadata['title']}".strip())
    if metadata.get("chapter"):
        parts.append(f"Capitolul {metadata.get('chapter_no', '')} - {metadata['chapter']}".strip())
    if metadata.get("article"):
        parts.append(f"Articolul {metadata.get('article_no', '')} - {metadata['article']}".strip())
    return " | ".join(parts) or metadata.get("headings", "sursa necunoscuta")


def make_retriever_node(store_name: str = VECTOR_DB_DIR, top_k: int = RETRIEVER_TOP_K):
    def retriever(state: AgentState) -> dict:
        store = load_vectorstore(store_name)
        results = store.similarity_search(state["question"], k=top_k)
        citations = [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]
        return {"citations": citations}

    return retriever


def make_generator_node():
    def generator(state: AgentState) -> dict:
        llm = get_llm()
        context = "\n\n".join(
            f"[{_format_citation_label(c['metadata'])}]\n{c['text']}" for c in state.get("citations", [])
        )
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context:\n{context}\n\nIntrebare: {state['question']}"),
        ]
        response = llm.invoke(messages)
        return {"answer": response.content}

    return generator


def build_graph(store_name: str = VECTOR_DB_DIR, top_k: int = RETRIEVER_TOP_K):
    """Build and compile the START -> Retriever -> Generator -> END graph."""
    graph = StateGraph(AgentState)
    graph.add_node("Retriever", make_retriever_node(store_name, top_k))
    graph.add_node("Generator", make_generator_node())
    graph.add_edge(START, "Retriever")
    graph.add_edge("Retriever", "Generator")
    graph.add_edge("Generator", END)
    return graph.compile()
