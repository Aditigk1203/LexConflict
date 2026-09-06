import json
from collections import defaultdict
from pathlib import Path


RISK_PATH = Path(
    "data/processed/risk/dev_risk_scores.json"
)

OUTPUT_PATH = Path(
    "data/processed/risk/dev_document_risk.json"
)


def extract_document_id(node_id):

    parts = str(node_id).split("_clause_")

    if len(parts) == 2:
        return parts[0]

    return str(node_id).split("_")[0]


def classify_document_risk(score):

    if score >= 0.80:
        return "critical"

    if score >= 0.60:
        return "high"

    if score >= 0.35:
        return "medium"

    if score >= 0.15:
        return "low"

    return "minimal"


def main():

    print("=" * 70)
    print("LEXCONFLICT DOCUMENT RISK")
    print("=" * 70)

    with open(
        RISK_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        risk_results = json.load(file)

    documents = defaultdict(list)

    for result in risk_results:

        document_id = extract_document_id(
            result["node_id"]
        )

        documents[
            document_id
        ].append(result)

    document_results = []

    for document_id, clauses in documents.items():

        scores = [
            clause["risk_score"]
            for clause in clauses
        ]

        # Maximum risk captures the most serious
        # unresolved issue.
        max_score = max(scores)

        # Mean captures overall contract pressure.
        average_score = (
            sum(scores)
            / len(scores)
        )

        high_count = sum(
            clause["risk_level"] in {
                "high",
                "critical",
            }
            for clause in clauses
        )

        conflict_count = sum(
            clause["conflict_degree"] > 0
            for clause in clauses
        )

        document_result = {
            "document_id": document_id,
            "number_of_clauses": len(clauses),
            "maximum_risk": round(
                max_score,
                4
            ),
            "average_risk": round(
                average_score,
                4
            ),
            "risk_level": classify_document_risk(
                max_score
            ),
            "high_risk_clause_count": high_count,
            "conflict_connected_clause_count": (
                conflict_count
            ),
        }

        document_results.append(
            document_result
        )

    document_results.sort(
        key=lambda x: x["maximum_risk"],
        reverse=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            document_results,
            file,
            indent=2
        )

    print()
    print("DOCUMENT RISK RESULTS")
    print("-" * 70)

    for result in document_results:

        print(
            f"Document {result['document_id']} | "
            f"Risk={result['risk_level']} | "
            f"Max={result['maximum_risk']} | "
            f"High-risk clauses="
            f"{result['high_risk_clause_count']}"
        )

    print()
    print(
        "Saved to:",
        OUTPUT_PATH
    )

    print()
    print("✓ DOCUMENT RISK ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()