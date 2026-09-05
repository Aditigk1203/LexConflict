import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


def analyze_file(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        clauses = json.load(file)

    modalities = Counter(
        clause.get("modality")
        for clause in clauses
    )

    parties = Counter(
        clause.get("party")
        for clause in clauses
    )

    conditions = sum(
        clause.get("condition") is not None
        for clause in clauses
    )

    print("\n" + "=" * 60)
    print(path.name)
    print("=" * 60)

    print(
        "Total clauses:",
        len(clauses)
    )

    print(
        "\nModality distribution:"
    )

    for key, value in modalities.items():

        print(
            f"  {key}: {value}"
        )

    print(
        "\nParty distribution:"
    )

    for key, value in parties.items():

        print(
            f"  {key}: {value}"
        )

    print(
        "\nClauses containing conditions:",
        conditions
    )


def main():

    for path in sorted(
        DATA_DIR.glob("*_clauses.json")
    ):

        analyze_file(path)


if __name__ == "__main__":
    main()