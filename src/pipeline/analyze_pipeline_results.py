import json
from pathlib import Path


RESULT_PATH = Path(
    "data/processed/pipeline/dev_pipeline_results.json"
)


def main():

    print("=" * 80)
    print("LEXCONFLICT PIPELINE RESULT ANALYSIS")
    print("=" * 80)

    if not RESULT_PATH.exists():
        print(f"ERROR: Result file not found: {RESULT_PATH}")
        return

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ================================================================
    # DATASET SUMMARY
    # ================================================================

    dataset = data.get("dataset", {})

    print("\n" + "-" * 80)
    print("PIPELINE SUMMARY")
    print("-" * 80)

    print(
        f"Documents available: "
        f"{dataset.get('documents_available', 'N/A')}"
    )

    print(
        f"Documents processed: "
        f"{dataset.get('documents_processed', 'N/A')}"
    )

    print(
        f"Clauses processed: "
        f"{dataset.get('clauses_processed', 'N/A')}"
    )

    print(
        f"Candidate pairs: "
        f"{dataset.get('candidate_pairs', 'N/A')}"
    )

    print(
        f"Conflict pairs: "
        f"{dataset.get('conflict_pairs', 'N/A')}"
    )

    # ================================================================
    # CONFIGURATION
    # ================================================================

    configuration = data.get("configuration", {})

    print("\n" + "-" * 80)
    print("PIPELINE CONFIGURATION")
    print("-" * 80)

    for key, value in configuration.items():
        print(f"{key}: {value}")

    # ================================================================
    # HYBRID RESULTS
    # ================================================================

    hybrid_results = data.get(
        "hybrid_results",
        []
    )

    conflicts = [
        result
        for result in hybrid_results
        if result.get("is_conflict", False)
    ]

    print("\n" + "-" * 80)
    print("CONFLICT DETECTION SUMMARY")
    print("-" * 80)

    print(f"Hybrid results: {len(hybrid_results)}")
    print(f"Detected conflicts: {len(conflicts)}")

    # ================================================================
    # ALL PAIRS SORTED BY HYBRID SCORE
    # ================================================================

    print("\n" + "-" * 80)
    print("TOP 15 PAIRS BY HYBRID CONFLICT SCORE")
    print("-" * 80)

    sorted_hybrid = sorted(
        hybrid_results,
        key=lambda x: x.get(
            "hybrid_conflict_score",
            0.0
        ),
        reverse=True
    )

    print(
        f"{'Rank':<6}"
        f"{'Query':<20}"
        f"{'Candidate':<20}"
        f"{'Retrieval':<11}"
        f"{'NLI Contr.':<12}"
        f"{'Struct.':<10}"
        f"{'Hybrid':<10}"
        f"{'Conflict'}"
    )

    print("-" * 100)

    for rank, result in enumerate(
        sorted_hybrid[:15],
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{str(result.get('query_clause_id'))[:19]:<20}"
            f"{str(result.get('candidate_clause_id'))[:19]:<20}"
            f"{result.get('retrieval_score', 0.0):<11.4f}"
            f"{result.get('nli_contradiction_probability', 0.0):<12.4f}"
            f"{result.get('structured_conflict_score', 0.0):<10.4f}"
            f"{result.get('hybrid_conflict_score', 0.0):<10.4f}"
            f"{'YES' if result.get('is_conflict') else 'NO'}"
        )

    # ================================================================
    # RETRIEVAL SCORE DISTRIBUTION
    # ================================================================

    print("\n" + "-" * 80)
    print("RETRIEVAL QUALITY CHECK")
    print("-" * 80)

    retrieval_scores = [
        float(
            result.get(
                "retrieval_score",
                0.0
            )
        )
        for result in hybrid_results
    ]

    if retrieval_scores:

        zero_count = sum(
            score == 0.0
            for score in retrieval_scores
        )

        positive_count = sum(
            score > 0.0
            for score in retrieval_scores
        )

        above_005 = sum(
            score >= 0.05
            for score in retrieval_scores
        )

        above_010 = sum(
            score >= 0.10
            for score in retrieval_scores
        )

        above_015 = sum(
            score >= 0.15
            for score in retrieval_scores
        )

        print(
            f"Minimum retrieval score: "
            f"{min(retrieval_scores):.4f}"
        )

        print(
            f"Maximum retrieval score: "
            f"{max(retrieval_scores):.4f}"
        )

        print(
            f"Mean retrieval score: "
            f"{sum(retrieval_scores) / len(retrieval_scores):.4f}"
        )

        print(
            f"Score = 0.00: "
            f"{zero_count}"
        )

        print(
            f"Score > 0.00: "
            f"{positive_count}"
        )

        print(
            f"Score >= 0.05: "
            f"{above_005}"
        )

        print(
            f"Score >= 0.10: "
            f"{above_010}"
        )

        print(
            f"Score >= 0.15: "
            f"{above_015}"
        )

    # ================================================================
    # DETECTED CONFLICTS
    # ================================================================

    print("\n" + "-" * 80)
    print("DETECTED CONFLICTS")
    print("-" * 80)

    if not conflicts:

        print("No conflicts detected.")

    else:

        for i, result in enumerate(
            conflicts,
            start=1
        ):

            print(
                f"\nConflict #{i}"
            )

            print("-" * 60)

            print(
                f"Query clause ID: "
                f"{result.get('query_clause_id')}"
            )

            print(
                f"Candidate clause ID: "
                f"{result.get('candidate_clause_id')}"
            )

            print(
                f"Document ID: "
                f"{result.get('document_id')}"
            )

            print(
                f"Retrieval rank: "
                f"{result.get('retrieval_rank')}"
            )

            print(
                f"Retrieval score: "
                f"{result.get('retrieval_score', 0.0):.4f}"
            )

            print(
                f"NLI label: "
                f"{result.get('nli_label')}"
            )

            print(
                f"NLI confidence: "
                f"{result.get('nli_confidence', 0.0):.4f}"
            )

            print(
                f"NLI contradiction probability: "
                f"{result.get('nli_contradiction_probability', 0.0):.4f}"
            )

            print(
                f"Structured conflict score: "
                f"{result.get('structured_conflict_score', 0.0):.4f}"
            )

            print(
                f"Structured conflict type: "
                f"{result.get('structured_conflict_type')}"
            )

            print(
                f"Hybrid conflict score: "
                f"{result.get('hybrid_conflict_score', 0.0):.4f}"
            )

            print(
                "\nHypothesis / query clause:"
            )

            print(
                result.get(
                    "hypothesis",
                    ""
                )
            )

            print(
                "\nCandidate clause:"
            )

            print(
                result.get(
                    "clause_text",
                    ""
                )
            )

            print(
                "\nExplanation:"
            )

            print(
                result.get(
                    "explanation",
                    ""
                )
            )

    # ================================================================
    # GRAPH SUMMARY
    # ================================================================

    graph = data.get(
        "graph",
        {}
    )

    print("\n" + "-" * 80)
    print("CONFLICT GRAPH SUMMARY")
    print("-" * 80)

    print(
        f"Graph nodes: "
        f"{graph.get('number_of_nodes', 'N/A')}"
    )

    print(
        f"Graph edges: "
        f"{graph.get('number_of_edges', 'N/A')}"
    )

    print(
        f"Conflict edges: "
        f"{graph.get('number_of_conflict_edges', 'N/A')}"
    )

    # ================================================================
    # PROPAGATION SUMMARY
    # ================================================================

    propagation_results = data.get(
        "propagation_results",
        {}
    )

    print("\n" + "-" * 80)
    print("GRAPH PROPAGATION SUMMARY")
    print("-" * 80)

    print(
        f"Propagation results: "
        f"{len(propagation_results)}"
    )

    if propagation_results:

        hotspot_counts = {}

        propagated_scores = []

        for node_id, result in propagation_results.items():

            level = result.get(
                "hotspot_level",
                "unknown"
            )

            hotspot_counts[level] = (
                hotspot_counts.get(level, 0) + 1
            )

            propagated_scores.append(
                float(
                    result.get(
                        "propagated_conflict_score",
                        0.0
                    )
                )
            )

        for level in [
            "high",
            "medium",
            "low",
            "none",
            "unknown"
        ]:

            if level in hotspot_counts:

                print(
                    f"{level}: "
                    f"{hotspot_counts[level]}"
                )

        if propagated_scores:

            print(
                f"Maximum propagated score: "
                f"{max(propagated_scores):.4f}"
            )

            print(
                f"Mean propagated score: "
                f"{sum(propagated_scores) / len(propagated_scores):.4f}"
            )

    # ================================================================
    # RISK SUMMARY
    # ================================================================

    risk_results = data.get(
        "risk_results",
        []
    )

    print("\n" + "-" * 80)
    print("RISK SUMMARY")
    print("-" * 80)

    print(
        f"Risk results: "
        f"{len(risk_results)}"
    )

    risk_levels = {}

    for result in risk_results:

        level = result.get(
            "risk_level",
            "unknown"
        )

        risk_levels[level] = (
            risk_levels.get(level, 0) + 1
        )

    for level in [
        "critical",
        "high",
        "medium",
        "low",
        "minimal",
        "unknown"
    ]:

        if level in risk_levels:

            print(
                f"{level}: "
                f"{risk_levels[level]}"
            )

    # ================================================================
    # TOP RISK CLAUSES
    # ================================================================

    sorted_risk = sorted(
        risk_results,
        key=lambda x: x.get(
            "risk_score",
            0.0
        ),
        reverse=True
    )

    print("\n" + "-" * 80)
    print("TOP 10 RISK CLAUSES")
    print("-" * 80)

    for i, result in enumerate(
        sorted_risk[:10],
        start=1
    ):

        print(
            f"{i}. "
            f"{result.get('node_id')} | "
            f"risk="
            f"{result.get('risk_score', 0.0):.4f} | "
            f"level="
            f"{result.get('risk_level')}"
        )

    # ================================================================
    # FINAL STATUS
    # ================================================================

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()