import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS = {
    "original_text": (
        PROJECT_ROOT
        / "results"
        / "retrieval"
        / "dev_tfidf_evidence_retrieval.json"
    ),
    "parent_context": (
        PROJECT_ROOT
        / "results"
        / "retrieval"
        / "dev_tfidf_context_evidence_retrieval.json"
    ),
}

THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
K_VALUES = [1, 2, 5, 10]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(query_results, threshold, k):
    hits = 0
    total_candidates_passed = 0
    queries_with_no_candidates = 0

    for result in query_results:
        gold_clause_ids = set(result["gold_clause_ids"])

        top_k = result["top_10"][:k]

        passed_candidates = [
            candidate
            for candidate in top_k
            if candidate["score"] >= threshold
        ]

        total_candidates_passed += len(passed_candidates)

        if not passed_candidates:
            queries_with_no_candidates += 1

        if any(
            candidate["clause_id"] in gold_clause_ids
            for candidate in passed_candidates
        ):
            hits += 1

    total_queries = len(query_results)

    return {
        "recall": hits / total_queries,
        "average_candidates_passed":
            total_candidates_passed / total_queries,
        "no_candidate_rate":
            queries_with_no_candidates / total_queries,
    }


def main():
    for report_name, report_path in REPORTS.items():
        report = load_json(report_path)
        query_results = report["query_results"]

        print("\n" + "=" * 72)
        print(f"THRESHOLD ANALYSIS: {report_name}")
        print("=" * 72)

        for k in K_VALUES:
            print(f"\nTOP_K = {k}")
            print(
                "threshold | recall@k | avg candidates | "
                "queries with none"
            )
            print("-" * 60)

            for threshold in THRESHOLDS:
                metrics = calculate_metrics(
                    query_results=query_results,
                    threshold=threshold,
                    k=k,
                )

                print(
                    f"{threshold:>9.2f} | "
                    f"{metrics['recall']:.4f}   | "
                    f"{metrics['average_candidates_passed']:.2f}           | "
                    f"{metrics['no_candidate_rate']:.4f}"
                )


if __name__ == "__main__":
    main()