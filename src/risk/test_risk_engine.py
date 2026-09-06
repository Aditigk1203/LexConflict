from src.risk import RiskEngine


def main():

    print("=" * 70)
    print("LEXCONFLICT RISK ENGINE TEST")
    print("=" * 70)

    propagation_data = {
        "nodes": [
            {
                "node_id": "A",
                "direct_score": 0.90,
                "propagated_score": 0.89,
                "conflict_degree": 2,
            },
            {
                "node_id": "B",
                "direct_score": 0.75,
                "propagated_score": 0.80,
                "conflict_degree": 1,
            },
            {
                "node_id": "C",
                "direct_score": 0.10,
                "propagated_score": 0.12,
                "conflict_degree": 0,
            },
        ]
    }

    graph_data = {
        "edges": [
            {
                "source_id": "A",
                "target_id": "B",
                "relationship": "conflict",
                "conflict_type": "obligation_vs_prohibition",
            },
            {
                "source_id": "A",
                "target_id": "C",
                "relationship": "conflict",
                "conflict_type": "positive_vs_negative",
            },
        ]
    }

    engine = RiskEngine()

    results = engine.calculate_graph_risk(
        propagation_data,
        graph_data
    )

    print()
    print("RISK RESULTS")
    print("-" * 70)

    for result in results:

        print()
        print("Node:", result.node_id)
        print(
            "Direct conflict:",
            round(
                result.direct_conflict_score,
                4
            )
        )
        print(
            "Propagated conflict:",
            round(
                result.propagated_conflict_score,
                4
            )
        )
        print(
            "Conflict degree:",
            result.conflict_degree
        )
        print(
            "Risk score:",
            round(
                result.risk_score,
                4
            )
        )
        print(
            "Risk level:",
            result.risk_level
        )
        print(
            "Explanation:",
            result.explanation
        )

    # ---------------------------------------------------------
    # Basic correctness checks
    # ---------------------------------------------------------

    assert len(results) == 3

    for result in results:

        assert 0.0 <= result.risk_score <= 1.0

        assert result.risk_level in {
            "minimal",
            "low",
            "medium",
            "high",
            "critical",
        }

    # A should be higher risk than C
    result_a = next(
        r for r in results
        if r.node_id == "A"
    )

    result_c = next(
        r for r in results
        if r.node_id == "C"
    )

    assert result_a.risk_score > result_c.risk_score

    print()
    print("✓ RISK ENGINE TEST PASSED")


if __name__ == "__main__":
    main()