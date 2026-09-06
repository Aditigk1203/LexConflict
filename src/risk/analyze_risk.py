import json
from pathlib import Path


RISK_PATH = Path(
    "data/processed/risk/dev_risk_scores.json"
)

SUMMARY_PATH = Path(
    "data/processed/risk/dev_risk_summary.json"
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    print("=" * 70)
    print("LEXCONFLICT RISK ANALYSIS")
    print("=" * 70)

    results = load_json(
        RISK_PATH
    )

    summary = load_json(
        SUMMARY_PATH
    )

    print()
    print("RISK STATISTICS")
    print("-" * 70)

    print(
        "Nodes:",
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
    print("-" * 70)

    for level, count in summary[
        "risk_level_counts"
    ].items():

        print(
            f"{level}: {count}"
        )

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    ranked = sorted(
        results,
        key=lambda x: x["risk_score"],
        reverse=True
    )

    print()
    print("TOP 10 HIGHEST-RISK CLAUSES")
    print("-" * 70)

    for index, result in enumerate(
        ranked[:10],
        start=1
    ):

        print()
        print(
            f"{index}. {result['node_id']}"
        )

        print(
            "Risk score:",
            round(
                result["risk_score"],
                4
            )
        )

        print(
            "Risk level:",
            result["risk_level"]
        )

        print(
            "Direct conflict:",
            round(
                result[
                    "direct_conflict_score"
                ],
                4
            )
        )

        print(
            "Propagated conflict:",
            round(
                result[
                    "propagated_conflict_score"
                ],
                4
            )
        )

        print(
            "Conflict degree:",
            result["conflict_degree"]
        )

        print(
            "Conflict density:",
            round(
                result["conflict_density"],
                4
            )
        )

        print(
            "Severity:",
            round(
                result["severity_score"],
                4
            )
        )

        print(
            "Explanation:",
            result["explanation"]
        )

    print()
    print("✓ RISK ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()