from evaluation import phoenix_evals


class _FakeScore:
    def __init__(self, label, score, explanation=""):
        self.label = label
        self.score = score
        self.explanation = explanation


class _FakeFaithfulnessEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, eval_input):
        assert "input" in eval_input and "output" in eval_input and "context" in eval_input
        return [_FakeScore("faithful", 1.0, "grounded in context")]


class _FakeRelevanceEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, eval_input):
        assert "input" in eval_input and "context" in eval_input
        return [_FakeScore("relevant", 0.0, "off topic")]


def test_evaluate_faithfulness_and_relevance_aggregates_scores(monkeypatch):
    monkeypatch.setattr("phoenix.evals.LLM", lambda **kwargs: object())
    monkeypatch.setattr("phoenix.evals.metrics.FaithfulnessEvaluator", _FakeFaithfulnessEvaluator)
    monkeypatch.setattr("phoenix.evals.metrics.RetrievalRelevanceEvaluator", _FakeRelevanceEvaluator)

    records = [
        {"question": "Ce este sediul permanent?", "answer": "Raspuns 1.", "citations": [{"text": "context 1"}]},
        {"question": "Cine plateste impozitul?", "answer": "Raspuns 2.", "citations": [{"text": "context 2"}]},
    ]

    report = phoenix_evals.evaluate_faithfulness_and_relevance(records)

    assert len(report["results"]) == 2
    assert report["results"][0]["faithfulness_label"] == "faithful"
    assert report["results"][0]["context_relevance_label"] == "relevant"
    assert report["summary"]["avg_faithfulness"] == 1.0
    assert report["summary"]["avg_context_relevance"] == 0.0
