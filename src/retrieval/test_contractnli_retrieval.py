import json
from pathlib import Path

from src.retrieval import CandidateRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_clauses.json"
)


def load_clauses():

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    print("=" * 70)
    print("CONTRACTNLI RETRIEVAL TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    clauses = load_clauses()

    print(
        "\nTotal clauses:",
        len(clauses)
    )

    if not clauses:

        print(
            "No clauses found."
        )

        return

    # --------------------------------------------------
    # Use only a subset initially
    # --------------------------------------------------

    # We start with 2,000 clauses so the first
    # test remains fast.

    test_clauses = clauses[:2000]

    print(
        "Clauses used for test:",
        len(test_clauses)
    )

    # --------------------------------------------------
    # Build retriever
    # --------------------------------------------------

    retriever = CandidateRetriever(
        top_k=5,
        same_document_only=True,
    )

    retriever.fit(
        test_clauses
    )

    print(
        "TF-IDF feature count:",
        retriever.representation.get_feature_count()
    )

    # --------------------------------------------------
    # Query first clause
    # --------------------------------------------------

    query = test_clauses[0]

    print("\n" + "=" * 70)

    print(
        "QUERY:"
    )

    print(
        query["clause_id"]
    )

    print(
        query["text"]
    )

    # --------------------------------------------------
    # Retrieve
    # --------------------------------------------------

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    print(
        "\nTOP CANDIDATES:"
    )

    for result in results:

        print(
            "\n"
            + "-" * 70
        )

        print(
            "Rank:",
            result.rank
        )

        print(
            "Candidate:",
            result.candidate_clause_id
        )

        print(
            "Similarity:",
            round(
                result.score,
                4
            )
        )

        print(
            "Text:",
            result.candidate_text
        )


if __name__ == "__main__":
    main()