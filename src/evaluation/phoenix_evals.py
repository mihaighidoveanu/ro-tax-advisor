"""Arize Phoenix LLM-judge evaluators (Faithfulness, Context Relevance) for RAG outputs.

Faithfulness checks whether an answer is grounded in its retrieved context (no
hallucination); Context Relevance ("Retrieval Relevance" in Phoenix's evals API) checks
whether the retrieved chunks actually help answer the question. Both require a real LLM
judge, so this module makes real API calls and is not part of the default lightweight
`evaluate.py` run.
"""
from config import MODEL_NAME


def _build_judge_llm(model_name: str = MODEL_NAME):
    from phoenix.evals import LLM

    return LLM(provider="openai", model=model_name)


def _context_text(citations) -> str:
    return "\n\n".join(c["text"] for c in citations)


def evaluate_faithfulness_and_relevance(records, model_name: str = MODEL_NAME) -> dict:
    """
    @param records list of {"question": str, "answer": str, "citations": [{"text": str, ...}]}
        (the shape returned by rag.rag.answer_question)
    @return {"results": [...], "summary": {"avg_faithfulness": float, "avg_context_relevance": float}}
    """
    from phoenix.evals.metrics import FaithfulnessEvaluator, RetrievalRelevanceEvaluator

    llm = _build_judge_llm(model_name)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevance_evaluator = RetrievalRelevanceEvaluator(llm=llm)

    results = []
    for record in records:
        context = _context_text(record["citations"])

        faithfulness_score = faithfulness_evaluator.evaluate(
            {"input": record["question"], "output": record["answer"], "context": context}
        )[0]
        relevance_score = relevance_evaluator.evaluate(
            {"input": record["question"], "context": context}
        )[0]

        results.append(
            {
                "question": record["question"],
                "faithfulness_label": faithfulness_score.label,
                "faithfulness_score": faithfulness_score.score,
                "faithfulness_explanation": faithfulness_score.explanation,
                "context_relevance_label": relevance_score.label,
                "context_relevance_score": relevance_score.score,
                "context_relevance_explanation": relevance_score.explanation,
            }
        )

    summary = {
        "avg_faithfulness": sum(r["faithfulness_score"] for r in results) / len(results),
        "avg_context_relevance": sum(r["context_relevance_score"] for r in results) / len(results),
    }
    return {"results": results, "summary": summary}


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from evaluation.dataset import EVAL_DATASET
    from rag.rag import answer_question

    records = []
    for item in EVAL_DATASET:
        outcome = answer_question(item["question"])
        records.append({"question": item["question"], "answer": outcome["answer"], "citations": outcome["citations"]})

    report = evaluate_faithfulness_and_relevance(records)
    for r in report["results"]:
        print(f"[{r['faithfulness_label']}/{r['context_relevance_label']}] {r['question']}")
    print(f"\nAvg faithfulness: {report['summary']['avg_faithfulness']:.0%}")
    print(f"Avg context relevance: {report['summary']['avg_context_relevance']:.0%}")
