import pandas as pd

from evaluation import ragas_benchmark
from rag import graph as graph_module


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def invoke(self, messages):
        return _FakeResponse("Raspuns generat.")


class _FakeMetricResult:
    def __init__(self, value):
        self.value = value


class _FakeMetric:
    def __init__(self, value):
        self._value = value

    def score(self, **kwargs):
        return _FakeMetricResult(self._value)


def test_load_indexed_chunks_returns_documents(fixture_vectorstore):
    documents = ragas_benchmark._load_indexed_chunks(fixture_vectorstore)

    assert len(documents) == 3
    assert all(doc.page_content for doc in documents)
    assert any(doc.metadata.get("article_no") == 8 for doc in documents)


def test_score_sample_shapes_all_four_metrics():
    metrics = {
        "faithfulness": _FakeMetric(1.0),
        "context_precision": _FakeMetric(0.8),
        "context_recall": _FakeMetric(0.6),
        "answer_relevancy": _FakeMetric(0.9),
    }

    scores = ragas_benchmark._score_sample(
        metrics, "Ce este sediul permanent?", "Raspuns.", ["context text"], "Referinta."
    )

    assert scores == {
        "faithfulness": 1.0,
        "context_precision": 0.8,
        "context_recall": 0.6,
        "answer_relevancy": 0.9,
    }


def test_run_ragas_benchmark_aggregates_summary(fixture_vectorstore, monkeypatch):
    monkeypatch.setattr(graph_module, "get_llm", lambda: _FakeLLM())
    monkeypatch.setattr(
        ragas_benchmark,
        "generate_testset",
        lambda store_name, testset_size: pd.DataFrame(
            [
                {"user_input": "Ce este sediul permanent?", "reference": "Referinta 1."},
                {"user_input": "Cine plateste impozitul pe profit?", "reference": "Referinta 2."},
            ]
        ),
    )
    monkeypatch.setattr(
        ragas_benchmark,
        "_build_ragas_metrics",
        lambda: {
            "faithfulness": _FakeMetric(1.0),
            "context_precision": _FakeMetric(1.0),
            "context_recall": _FakeMetric(1.0),
            "answer_relevancy": _FakeMetric(1.0),
        },
    )

    report = ragas_benchmark.run_ragas_benchmark(store_name=fixture_vectorstore, verbose=False)

    assert len(report["results"]) == 2
    assert report["summary"] == {
        "faithfulness": 1.0,
        "context_precision": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
    }
