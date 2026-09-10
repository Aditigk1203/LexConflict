import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval"
    / "dev_tfidf_evidence_retrieval.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval"
    / "dev_top2_retrieval_audit.csv"
)

TOP_K = 2
MAX_FAILED_QUERIES = 50


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def failure_sort_key(result):
    rank = result.get("first_relevant_rank")

    # Missing gold evidence from top 10 is worst.
    if rank is None:
        return 999999

    return rank


def main():
    report = load_json(REPORT_PATH)
    query_results = report["query_results"]

    failed_queries = [
        result
        for result in query_results
        if (
            result.get("first_relevant_rank") is None
            or result["first_relevant_rank"] > TOP_K
        )
    ]

    failed_queries = sorted(
        failed_queries,
        key=failure_sort_key,
        reverse=True,
    )[:MAX_FAILED_QUERIES]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "document_id",
        "hypothesis_id",
        "hypothesis",
        "first_relevant_rank",
        "gold_clause_ids",
        "candidate_rank",
        "candidate_clause_id",
        "candidate_score",
        "candidate_is_gold_evidence",
        "candidate_text",
        "human_relevance",
        "review_notes",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in failed_queries:
            gold_clause_ids = set(
                result["gold_clause_ids"]
            )

            for rank, candidate in enumerate(
                result["top_10"][:TOP_K],
                start=1,
            ):
                writer.writerow(
                    {
                        "document_id":
                            result["document_id"],
                        "hypothesis_id":
                            result["hypothesis_id"],
                        "hypothesis":
                            result["hypothesis"],
                        "first_relevant_rank":
                            result["first_relevant_rank"],
                        "gold_clause_ids":
                            " | ".join(
                                result["gold_clause_ids"]
                            ),
                        "candidate_rank": rank,
                        "candidate_clause_id":
                            candidate["clause_id"],
                        "candidate_score":
                            round(candidate["score"], 4),
                        "candidate_is_gold_evidence":
                            candidate["clause_id"]
                            in gold_clause_ids,
                        "candidate_text":
                            candidate["text"],
                        "human_relevance": "",
                        "review_notes": "",
                    }
                )

    print(
        f"Exported {len(failed_queries)} failed "
        f"TOP_K={TOP_K} queries."
    )
    print(f"Audit file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()