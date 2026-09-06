import json
from pathlib import Path


RISK_PATH = Path(
    "data/processed/risk/dev_risk_scores.json"
)

SUMMARY_PATH = Path(
    "data/processed/risk/dev_risk_summary.json"
)

DOCUMENT_PATH = Path(
    "data/processed/risk/dev_document_risk.json"
)


def main():

    print("=" * 70)
    print("LEXCONFLICT SAVED RISK VALIDATION")
    print("=" * 70)

    assert RISK_PATH.exists(), (
        "Risk score file not found."
    )

    assert SUMMARY_PATH.exists(), (
        "Risk summary file not found."
    )

    assert DOCUMENT_PATH.exists(), (
        "Document risk file not found."
    )

    with open(
        RISK_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        risks = json.load(file)

    with open(
        SUMMARY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        summary = json.load(file)

    with open(
        DOCUMENT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        document_risks = json.load(file)

    assert len(risks) > 0

    for result in risks:

        assert "node_id" in result
        assert "risk_score" in result
        assert "risk_level" in result
        assert "explanation" in result

        assert (
            0.0
            <= result["risk_score"]
            <= 1.0
        )

        assert result["risk_level"] in {
            "minimal",
            "low",
            "medium",
            "high",
            "critical",
        }

    assert (
        summary["number_of_nodes"]
        == len(risks)
    )

    assert len(document_risks) > 0

    print()
    print(
        "Nodes validated:",
        len(risks)
    )

    print(
        "Documents validated:",
        len(document_risks)
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
    print(
        "✓ SAVED RISK VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()