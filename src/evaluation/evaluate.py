import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.dataset import EVAL_DATASET
from monitoring.phoenix import init_tracing
from rag.rag import answer_question


def _retrieval_hit(citations, expected_article) -> bool:
    return any(c["metadata"].get("article_no") == expected_article for c in citations)


def _keyword_score(answer: str, expected_keywords) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def run_evaluation(store_name=None, dataset=EVAL_DATASET, verbose: bool = True, include_llm_judges: bool = False):
    """
    Run the RAG pipeline over the evaluation dataset and score retrieval accuracy
    (did the expected article get retrieved) and answer quality (keyword coverage).

    @param include_llm_judges Also score each answer with Phoenix's Faithfulness and
        Context Relevance LLM-judge evaluators (see phoenix_evals.py). Off by default:
        it makes real LLM calls (cost + latency), unlike the rest of this function.
    @return dict with "results" (per-question dicts) and "summary" (aggregate scores)
    """
    init_tracing()

    results = []
    for item in dataset:
        kwargs = {"store_name": store_name} if store_name else {}
        outcome = answer_question(item["question"], **kwargs)
        retrieval_hit = _retrieval_hit(outcome["citations"], item["expected_article"])
        keyword_score = _keyword_score(outcome["answer"], item["expected_keywords"])

        result = {
            "question": item["question"],
            "answer": outcome["answer"],
            "citations": outcome["citations"],
            "retrieval_hit": retrieval_hit,
            "keyword_score": keyword_score,
        }
        results.append(result)

        if verbose:
            status = "OK" if retrieval_hit else "MISS"
            print(f"[{status}] ({keyword_score:.0%} keywords) {item['question']}")

    summary = {
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in results) / len(results),
        "avg_keyword_score": sum(r["keyword_score"] for r in results) / len(results),
    }
    if verbose:
        print(f"\nRetrieval hit rate: {summary['retrieval_hit_rate']:.0%}")
        print(f"Average keyword score: {summary['avg_keyword_score']:.0%}")

    if include_llm_judges:
        from evaluation.phoenix_evals import evaluate_faithfulness_and_relevance

        judge_report = evaluate_faithfulness_and_relevance(results)
        for result, judge_result in zip(results, judge_report["results"]):
            result["faithfulness_label"] = judge_result["faithfulness_label"]
            result["context_relevance_label"] = judge_result["context_relevance_label"]
        summary.update(judge_report["summary"])

        if verbose:
            print(f"Avg faithfulness: {summary['avg_faithfulness']:.0%}")
            print(f"Avg context relevance: {summary['avg_context_relevance']:.0%}")

    return {"results": results, "summary": summary}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the hand-curated RAG evaluation set")
    parser.add_argument(
        "--llm-judges",
        action="store_true",
        help="Also score answers with Phoenix's Faithfulness and Context Relevance LLM judges (extra API calls)",
    )
    args = parser.parse_args()

    run_evaluation(include_llm_judges=args.llm_judges)
