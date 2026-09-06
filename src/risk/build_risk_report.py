import json
from pathlib import Path

from src.risk import RiskEngine


GRAPH_PATH = Path(
    "data/processed/conflict_graph/dev_conflict_graph.json"
)

PROPAGATION_PATH = Path(
    "data/processed/conflict_graph/dev_graph_propagation.json"
)

OUTPUT_DIR = Path(
    "data/processed/risk"
)

RISK_OUTPUT = OUTPUT_DIR / "dev_risk_scores.json"
SUMMARY_OUTPUT = OUTPUT_DIR / "dev_risk_summary.json"


def load_json(path):

    print(f"Loading: {path}")

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_json(data, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )


def main():

    print("=" * 70)
    print("LEXCONFLICT RISK SCORING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load existing graph
    # ---------------------------------------------------------

    graph_data = load_json(
        GRAPH_PATH
    )

    # ---------------------------------------------------------
    # Load existing graph propagation results
    # ---------------------------------------------------------

    propagation_data = load_json(
        PROPAGATION_PATH
    )

    print()
    print(
        "Graph nodes:",
        len(
            graph_data.get(
                "nodes",
                []
            )
        )
    )

    print(
        "Graph edges:",
        len(
            graph_data.get(
                "edges",
                []
            )
        )
    )

    # ---------------------------------------------------------
    # Create risk engine
    # ---------------------------------------------------------

    engine = RiskEngine(
        propagated_weight=0.50,
        direct_weight=0.30,
        connectivity_weight=0.20,
    )

    # ---------------------------------------------------------
    # Calculate risk
    # ---------------------------------------------------------

    results = engine.calculate_graph_risk(
        propagation_data=propagation_data,
        graph_data=graph_data,
    )

    # ---------------------------------------------------------
    # Convert to JSON
    # ---------------------------------------------------------

    risk_records = [
        result.to_dict()
        for result in results
    ]

    summary = engine.summarize_risk(
        results
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    save_json(
        risk_records,
        RISK_OUTPUT
    )

    save_json(
        summary,
        SUMMARY_OUTPUT
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RISK SCORING COMPLETE")
    print("=" * 70)

    print(
        "Number of nodes:",
        summary["number_of_nodes"]
    )

    print(
        "Average risk:",
        summary["average_risk"]
    )

    print(
        "Maximum risk:",
        summary["maximum_risk"]
    )

    print(
        "Overall risk:",
        summary["overall_risk_level"]
    )

    print()
    print("RISK LEVEL COUNTS")

    for level, count in summary[
        "risk_level_counts"
    ].items():

        print(
            f"{level}: {count}"
        )

    print()
    print("TOP RISK NODES")

    for item in summary[
        "high_risk_nodes"
    ]:

        print(
            f"{item['node_id']} | "
            f"score={item['risk_score']} | "
            f"level={item['risk_level']} | "
            f"degree={item['conflict_degree']}"
        )

    print()
    print("Risk scores saved to:")
    print(RISK_OUTPUT)

    print()
    print("Summary saved to:")
    print(SUMMARY_OUTPUT)

    print()
    print("✓ RISK SCORING COMPLETE")


if __name__ == "__main__":
    main()