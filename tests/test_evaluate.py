from evaluation import phoenix_evals
from evaluation.evaluate import _keyword_score, _retrieval_hit, run_evaluation
from rag import graph as graph_module


def test_retrieval_hit_true_when_article_present():
    citations = [{"metadata": {"article_no": 8}}, {"metadata": {"article_no": 21}}]
    assert _retrieval_hit(citations, 8) is True


def test_retrieval_hit_false_when_article_missing():
    citations = [{"metadata": {"article_no": 21}}]
    assert _retrieval_hit(citations, 8) is False


def test_keyword_score_full_and_partial_match():
    assert _keyword_score("sediul permanent este un loc", ["sediul", "permanent"]) == 1.0
    assert _keyword_score("sediul este un loc", ["sediul", "permanent"]) == 0.5
    assert _keyword_score("text irelevant", ["sediul", "permanent"]) == 0.0


def test_keyword_score_empty_keywords_returns_full_score():
    assert _keyword_score("orice raspuns", []) == 1.0


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def invoke(self, messages):
        return _FakeResponse("Sediul permanent este definit la articolul 8.")


_TINY_DATASET = [
    {"question": "Ce este sediul permanent?", "expected_keywords": ["sediul", "permanent"], "expected_article": 8},
]


def test_run_evaluation_without_llm_judges(fixture_vectorstore, monkeypatch):
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM())

    report = run_evaluation(store_name=fixture_vectorstore, dataset=_TINY_DATASET, verbose=False)

    assert report["summary"]["retrieval_hit_rate"] == 1.0
    assert "avg_faithfulness" not in report["summary"]
    assert "citations" in report["results"][0]


def test_run_evaluation_with_llm_judges_merges_summary(fixture_vectorstore, monkeypatch):
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM())
    monkeypatch.setattr(
        phoenix_evals,
        "evaluate_faithfulness_and_relevance",
        lambda records: {
            "results": [
                {"faithfulness_label": "faithful", "context_relevance_label": "relevant"} for _ in records
            ],
            "summary": {"avg_faithfulness": 1.0, "avg_context_relevance": 1.0},
        },
    )

    report = run_evaluation(store_name=fixture_vectorstore, dataset=_TINY_DATASET, verbose=False, include_llm_judges=True)

    assert report["summary"]["avg_faithfulness"] == 1.0
    assert report["summary"]["avg_context_relevance"] == 1.0
    assert report["results"][0]["faithfulness_label"] == "faithful"
