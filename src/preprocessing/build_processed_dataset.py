import json
from pathlib import Path

from src.preprocessing import preprocess_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "kaggle"
    / "contract-nli"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


def process_contract_nli_file(
    filename: str
):

    input_path = INPUT_DIR / filename

    output_path = (
        OUTPUT_DIR
        / filename.replace(
            ".json",
            "_clauses.json"
        )
    )

    print(f"\nProcessing {filename}...")

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    documents = data.get(
        "documents",
        []
    )

    all_clauses = []

    for document in documents:

        clauses = preprocess_document(
            document,
            dataset="contractnli"
        )

        for clause in clauses:

            all_clauses.append(
                clause.to_dict()
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_clauses,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved {len(all_clauses)} clauses"
    )

    print(
        f"Output: {output_path}"
    )


def main():

    for filename in [
        "train.json",
        "dev.json",
        "test.json"
    ]:

        process_contract_nli_file(
            filename
        )


if __name__ == "__main__":
    main()