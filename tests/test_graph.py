from rag import graph as graph_module
from rag.graph import build_graph, make_retriever_node


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content="Raspuns simulat."):
        self.content = content
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return _FakeResponse(self.content)


def test_retriever_node_populates_citations(fixture_vectorstore):
    retriever = make_retriever_node(store_name=fixture_vectorstore, top_k=2)

    out = retriever({"question": "Ce este sediul permanent?"})

    assert len(out["citations"]) == 2
    assert out["citations"][0]["metadata"].get("article_no") == 8


def test_generator_node_uses_citations_as_context(monkeypatch):
    fake_llm = _FakeLLM("Sediul permanent este definit la articolul 8.")
    monkeypatch.setattr(graph_module, "get_llm", lambda: fake_llm)

    generator = graph_module.make_generator_node()
    state = {
        "question": "Ce este sediul permanent?",
        "citations": [{"text": "Sediul permanent este...", "metadata": {"article": "Definitia sediului permanent", "article_no": 8}}],
    }

    out = generator(state)

    assert out["answer"] == fake_llm.content
    context_message = fake_llm.last_messages[1][1]
    assert "Sediul permanent este..." in context_message
    assert "Articolul 8" in context_message


def test_build_graph_end_to_end(fixture_vectorstore, monkeypatch):
    fake_llm = _FakeLLM("Raspuns final.")
    monkeypatch.setattr(graph_module, "get_llm", lambda: fake_llm)

    compiled = build_graph(store_name=fixture_vectorstore, top_k=2)
    result = compiled.invoke({"question": "Ce este sediul permanent?"})

    assert result["answer"] == "Raspuns final."
    assert len(result["citations"]) == 2
