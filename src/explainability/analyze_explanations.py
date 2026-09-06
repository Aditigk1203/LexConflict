import json
from pathlib import Path

from .explainability_engine import ExplainabilityEngine


def main():

    print("=" * 70)
    print("LEXCONFLICT EXPLAINABLE AI ANALYSIS")
    print("=" * 70)

    risk_path = (
        "data/processed/risk/dev_risk_scores.json"
    )

    graph_path = (
        "data/processed/conflict_graph/dev_conflict_graph.json"
    )

    output_dir = Path(
        "data/processed/explainability"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    engine = ExplainabilityEngine(
        risk_scores_path=risk_path,
        graph_path=graph_path
    )

    explanations = engine.explain_all()

    # ---------------------------------------------------------
    # Sort by risk
    # ---------------------------------------------------------

    explanations = sorted(
        explanations,
        key=lambda x: x.risk_score,
        reverse=True
    )

    # ---------------------------------------------------------
    # Print top 10
    # ---------------------------------------------------------

    print("\nTOP 10 EXPLAINED RISK CLAUSES")
    print("-" * 70)

    for index, result in enumerate(
        explanations[:10],
        start=1
    ):

        print(
            f"\n{index}. {result.node_id}"
        )

        print(
            f"Risk score: {result.risk_score:.4f}"
        )

        print(
            f"Risk level: {result.risk_level}"
        )

        print(
            f"Direct conflict: "
            f"{result.direct_conflict_score:.4f}"
        )

        print(
            f"Propagated conflict: "
            f"{result.propagated_conflict_score:.4f}"
        )

        print(
            f"Conflict degree: "
            f"{result.conflict_degree}"
        )

        print(
            f"Conflict types: "
            f"{result.conflict_types}"
        )

        print(
            "Explanation:"
        )

        print(
            result.explanation
        )

        print(
            "Recommendation:"
        )

        print(
            result.recommendation
        )

    # ---------------------------------------------------------
    # Save explanations
    # ---------------------------------------------------------

    explanation_path = (
        output_dir /
        "dev_explanations.json"
    )

    with open(
        explanation_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            [
                result.to_dict()
                for result in explanations
            ],
            f,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    risk_counts = {}

    for result in explanations:

        risk_counts[result.risk_level] = (
            risk_counts.get(
                result.risk_level,
                0
            ) + 1
        )

    summary = {
        "number_of_explanations": len(
            explanations
        ),
        "risk_level_counts": risk_counts,
        "top_risk_nodes": [
            {
                "node_id": result.node_id,
                "risk_score": result.risk_score,
                "risk_level": result.risk_level
            }
            for result in explanations[:10]
        ]
    }

    summary_path = (
        output_dir /
        "dev_explanation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print("\n")
    print(
        "Explanations saved to:"
    )

    print(
        explanation_path
    )

    print(
        "Summary saved to:"
    )

    print(
        summary_path
    )

    print("\n✓ EXPLAINABLE AI ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()