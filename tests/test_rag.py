from rag import graph as graph_module
from rag.rag import answer_question


class _FakeResponse:
    def __init__(self, content):
        self.content = content


def test_answer_question_returns_answer_and_citations(fixture_vectorstore, monkeypatch):
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM())

    result = answer_question("Ce este sediul permanent?", store_name=fixture_vectorstore)

    assert result["answer"] == "Raspuns simulat despre sediul permanent."
    assert len(result["citations"]) > 0
    assert result["citations"][0]["metadata"].get("article_no") == 8


class _FakeLLM:
    def invoke(self, messages):
        return _FakeResponse("Raspuns simulat despre sediul permanent.")
