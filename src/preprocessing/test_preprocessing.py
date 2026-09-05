import json
from pathlib import Path

from src.preprocessing.clause_preprocessor import preprocess_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTRACT_NLI_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "kaggle"
    / "contract-nli"
    / "train.json"
)


def main():

    print("=" * 60)
    print("LEXCONFLICT PREPROCESSING TEST")
    print("=" * 60)

    with open(
        CONTRACT_NLI_TRAIN,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    documents = data.get(
        "documents",
        []
    )

    print(
        f"Documents loaded: {len(documents)}"
    )

    if not documents:

        print("No documents found.")
        return

    # Process only ONE document first.
    # We don't need to process the whole dataset yet.

    document = documents[0]

    clauses = preprocess_document(
        document,
        dataset="contractnli"
    )

    print(
        f"Clauses generated: {len(clauses)}"
    )

    print("\nFirst 5 clauses:\n")

    for clause in clauses[:5]:

        print("-" * 60)

        print(
            "Document ID:",
            clause.document_id
        )

        print(
            "Clause ID:",
            clause.clause_id
        )

        print(
            "Text:",
            clause.text
        )

        print(
            "Party:",
            clause.party
        )

        print(
            "Modality:",
            clause.modality
        )

        print(
            "Condition:",
            clause.condition
        )

        print(
            "Proposition:",
            clause.proposition
        )

        print(
            "Dataset:",
            clause.dataset
        )


if __name__ == "__main__":
    main()