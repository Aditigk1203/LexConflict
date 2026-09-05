import json
from pathlib import Path

from src.retrieval.candidate_retrieval import CandidateRetriever
from src.reasoning.conflict_engine import ConflictEngine


def load_clauses(json_path):
    """
    Load processed clauses from a JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_clause(clause):
    """
    Print useful information about a clause.
    """
    print(f"Clause ID : {clause.get('clause_id')}")
    print(f"Document  : {clause.get('document_id')}")
    print(f"Party     : {clause.get('party')}")
    print(f"Modality  : {clause.get('modality')}")
    print(f"Condition : {clause.get('condition')}")
    print(f"Text      : {clause.get('text')}")


def main():

    # ---------------------------------------------------------
    # 1. Locate processed ContractNLI data
    # ---------------------------------------------------------

    data_path = Path("data/processed/test_clauses.json")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}"
        )

    print("=" * 70)
    print("LEXCONFLICT - RETRIEVAL + CONFLICT ENGINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 2. Load clauses
    # ---------------------------------------------------------

    clauses = load_clauses(data_path)

    print(f"\nLoaded clauses: {len(clauses)}")

    # ---------------------------------------------------------
    # 3. Use a small subset for the first integration test
    # ---------------------------------------------------------

    # We deliberately use a small number first.
    # This prevents unnecessary computation while testing
    # that the modules work together correctly.

    test_clauses = clauses[:500]

    print(f"Clauses used for integration test: {len(test_clauses)}")

    # ---------------------------------------------------------
    # 4. Create candidate retriever
    # ---------------------------------------------------------

    retriever = CandidateRetriever(
        top_k=5,
        same_document_only=True
    )

    print("\nFitting TF-IDF retrieval model...")

    retriever.fit(test_clauses)

    print("Retrieval model fitted successfully.")

    # ---------------------------------------------------------
    # 5. Create conflict engine
    # ---------------------------------------------------------

    conflict_engine = ConflictEngine()

    # ---------------------------------------------------------
    # 6. Select a query clause
    # ---------------------------------------------------------

    query_clause = test_clauses[0]

    print("\n" + "=" * 70)
    print("QUERY CLAUSE")
    print("=" * 70)

    print_clause(query_clause)

    # ---------------------------------------------------------
    # 7. Retrieve candidate clauses
    # ---------------------------------------------------------

    results = retriever.retrieve(query_clause)

    print("\n" + "=" * 70)
    print("RETRIEVED CANDIDATES")
    print("=" * 70)

    print(f"Number of candidates: {len(results)}")

    # ---------------------------------------------------------
    # 8. Analyze each retrieved pair
    # ---------------------------------------------------------

    conflict_results = []

    for result in results:

        candidate_clause = next(
            (
                clause
                for clause in test_clauses
                if clause.get("clause_id") == result.candidate_clause_id
            ),
            None
        )

        if candidate_clause is None:
            continue

        conflict_result = conflict_engine.analyze_pair(
            query_clause,
            candidate_clause
        )

        conflict_results.append(
            {
                "query_clause_id": result.query_clause_id,
                "candidate_clause_id": result.candidate_clause_id,
                "retrieval_score": result.score,
                "rank": result.rank,
                "is_conflict": conflict_result.is_conflict,
                "confidence": conflict_result.confidence,
                "conflict_type": conflict_result.conflict_type,
                "semantic_similarity": (
                    conflict_result.semantic_similarity
                ),
                "modality_conflict": (
                    conflict_result.modality_conflict
                ),
                "negation_conflict": (
                    conflict_result.negation_conflict
                ),
                "same_party": (
                    conflict_result.same_party
                ),
                "condition_similarity": (
                    conflict_result.condition_similarity
                ),
                "explanation": conflict_result.explanation
            }
        )

    # ---------------------------------------------------------
    # 9. Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFLICT ANALYSIS")
    print("=" * 70)

    for item in conflict_results:

        print("\n" + "-" * 70)

        print(
            f"Rank: {item['rank']} | "
            f"Retrieval score: {item['retrieval_score']:.4f}"
        )

        print(
            f"Candidate clause: "
            f"{item['candidate_clause_id']}"
        )

        print(
            f"Conflict: "
            f"{item['is_conflict']}"
        )

        print(
            f"Confidence: "
            f"{item['confidence']:.4f}"
        )

        print(
            f"Conflict type: "
            f"{item['conflict_type']}"
        )

        print(
            f"Semantic similarity: "
            f"{item['semantic_similarity']:.4f}"
        )

        print(
            f"Modality conflict: "
            f"{item['modality_conflict']}"
        )

        print(
            f"Negation conflict: "
            f"{item['negation_conflict']}"
        )

        print(
            f"Same party: "
            f"{item['same_party']}"
        )

        print(
            f"Condition similarity: "
            f"{item['condition_similarity']:.4f}"
        )

        print(
            f"Explanation: "
            f"{item['explanation']}"
        )

    # ---------------------------------------------------------
    # 10. Summary
    # ---------------------------------------------------------

    conflict_count = sum(
        1
        for item in conflict_results
        if item["is_conflict"]
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Query clause: {query_clause.get('clause_id')}")
    print(f"Candidates analyzed: {len(conflict_results)}")
    print(f"Potential conflicts: {conflict_count}")

    print("\nIntegration test completed successfully.")


if __name__ == "__main__":
    main()