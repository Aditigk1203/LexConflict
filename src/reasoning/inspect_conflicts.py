import json
from pathlib import Path


RESULTS_PATH = Path(
    "results/reasoning/conflict_detection.json"
)

CLAUSES_PATH = Path(
    "data/processed/dev_clauses.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 80)
    print("LEXCONFLICT - DETECTED CONFLICT INSPECTION")
    print("=" * 80)

    # ------------------------------------------------------------
    # LOAD RESULTS
    # ------------------------------------------------------------

    results = load_json(RESULTS_PATH)

    # ------------------------------------------------------------
    # LOAD CLAUSES
    # ------------------------------------------------------------

    clauses = load_json(CLAUSES_PATH)

    clause_map = {
        clause.get("clause_id"): clause
        for clause in clauses
    }

    # ------------------------------------------------------------
    # FILTER CONFLICTS
    # ------------------------------------------------------------

    conflicts = [
        r for r in results
        if isinstance(r, dict)
        and r.get("is_conflict") is True
    ]

    print(f"\nTotal result entries: {len(results)}")
    print(f"Detected conflicts: {len(conflicts)}")

    print("\n" + "=" * 80)
    print("CONFLICTING CLAUSE PAIRS")
    print("=" * 80)

    # ------------------------------------------------------------
    # DISPLAY CONFLICTS
    # ------------------------------------------------------------

    for i, conflict in enumerate(conflicts, start=1):

        query_id = conflict.get("query_clause_id")
        candidate_id = conflict.get("candidate_clause_id")

        query_clause = clause_map.get(query_id, {})
        candidate_clause = clause_map.get(candidate_id, {})

        query_text = query_clause.get(
            "text",
            "[CLAUSE TEXT NOT FOUND]"
        )

        candidate_text = candidate_clause.get(
            "text",
            "[CLAUSE TEXT NOT FOUND]"
        )

        print("\n" + "-" * 80)
        print(f"CONFLICT #{i}")
        print("-" * 80)

        print(f"\nQuery clause ID: {query_id}")
        print("\nQuery clause:")
        print(query_text)

        print(f"\nCandidate clause ID: {candidate_id}")
        print("\nCandidate clause:")
        print(candidate_text)

        print("\n" + "-" * 80)
        print("REASONING")
        print("-" * 80)

        print(
            f"Hybrid conflict score: "
            f"{conflict.get('hybrid_conflict_score', 0):.4f}"
        )

        print(
            f"Structured conflict score: "
            f"{conflict.get('structured_conflict_score', 0):.4f}"
        )

        print(
            f"NLI label: "
            f"{conflict.get('nli_label')}"
        )

        print(
            f"NLI contradiction probability: "
            f"{conflict.get('nli_contradiction_probability', 0):.4f}"
        )

        print(
            f"Conflict type: "
            f"{conflict.get('structured_conflict_type')}"
        )

        print(
            f"Semantic similarity: "
            f"{conflict.get('semantic_similarity', 0):.4f}"
        )

        print(
            f"Modality conflict: "
            f"{conflict.get('modality_conflict', 0):.4f}"
        )

        print(
            f"Negation conflict: "
            f"{conflict.get('negation_conflict', 0):.4f}"
        )

        print(
            f"Same party: "
            f"{conflict.get('same_party')}"
        )

        print(
            f"Condition similarity: "
            f"{conflict.get('condition_similarity', 0):.4f}"
        )

        print(
            f"Confidence level: "
            f"{conflict.get('confidence_level')}"
        )

        print("\nExplanation:")
        print(
            conflict.get(
                "explanation",
                "No explanation available."
            )
        )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CONFLICT TYPE SUMMARY")
    print("=" * 80)

    type_counts = {}

    for conflict in conflicts:

        conflict_type = conflict.get(
            "structured_conflict_type",
            "unknown"
        )

        type_counts[conflict_type] = (
            type_counts.get(conflict_type, 0) + 1
        )

    for conflict_type, count in sorted(
        type_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(f"{conflict_type}: {count}")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()