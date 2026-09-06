from .explainability_engine import ExplainabilityEngine


def main():

    print("=" * 70)
    print("LEXCONFLICT EXPLAINABILITY TEST")
    print("=" * 70)

    risk_path = (
        "data/processed/risk/dev_risk_scores.json"
    )

    graph_path = (
        "data/processed/conflict_graph/dev_conflict_graph.json"
    )

    engine = ExplainabilityEngine(
        risk_scores_path=risk_path,
        graph_path=graph_path
    )

    # ---------------------------------------------------------
    # Test 1: Highest-risk clause
    # ---------------------------------------------------------

    print("\nTEST 1 - HIGH RISK CLAUSE")
    print("-" * 70)

    explanation = engine.explain_clause(
        "7_clause_0015"
    )

    print(
        "Node:",
        explanation.node_id
    )

    print(
        "Risk score:",
        explanation.risk_score
    )

    print(
        "Risk level:",
        explanation.risk_level
    )

    print(
        "Direct conflict:",
        explanation.direct_conflict_score
    )

    print(
        "Propagated conflict:",
        explanation.propagated_conflict_score
    )

    print(
        "Conflict degree:",
        explanation.conflict_degree
    )

    print(
        "Conflict types:",
        explanation.conflict_types
    )

    print(
        "Neighbors:",
        explanation.conflict_neighbors
    )

    print(
        "\nExplanation:"
    )

    print(
        explanation.explanation
    )

    print(
        "\nRecommendation:"
    )

    print(
        explanation.recommendation
    )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert explanation.node_id == "7_clause_0015"

    assert 0.0 <= explanation.risk_score <= 1.0

    assert explanation.risk_level in [
        "minimal",
        "medium",
        "high",
        "critical"
    ]

    assert 0.0 <= explanation.direct_conflict_score <= 1.0

    assert 0.0 <= explanation.propagated_conflict_score <= 1.0

    assert explanation.conflict_degree >= 0

    assert explanation.conflict_density >= 0

    assert explanation.severity_score >= 0

    assert isinstance(
        explanation.explanation,
        str
    )

    assert len(
        explanation.explanation
    ) > 0

    assert isinstance(
        explanation.recommendation,
        str
    )

    # ---------------------------------------------------------
    # Test 2: Minimal-risk clause
    # ---------------------------------------------------------

    print("\n")
    print("TEST 2 - MINIMAL RISK CLAUSE")
    print("-" * 70)

    minimal_explanation = engine.explain_clause(
        "3_clause_0004"
    )

    print(
        "Node:",
        minimal_explanation.node_id
    )

    print(
        "Risk score:",
        minimal_explanation.risk_score
    )

    print(
        "Risk level:",
        minimal_explanation.risk_level
    )

    print(
        "Explanation:"
    )

    print(
        minimal_explanation.explanation
    )

    assert minimal_explanation.risk_level == "minimal"

    assert minimal_explanation.risk_score == 0.0

    # ---------------------------------------------------------
    # Test 3: Explain all clauses
    # ---------------------------------------------------------

    print("\n")
    print("TEST 3 - ALL CLAUSES")
    print("-" * 70)

    all_explanations = engine.explain_all()

    print(
        "Number of explanations:",
        len(all_explanations)
    )

    assert len(all_explanations) == len(
        engine.risk_scores
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n")
    print("✓ EXPLAINABILITY TEST PASSED")


if __name__ == "__main__":
    main()