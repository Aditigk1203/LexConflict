import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTRACT_NLI_DIR = (
    PROJECT_ROOT / "data" / "kaggle" / "contract-nli"
)

CUAD_DIR = (
    PROJECT_ROOT / "data" / "kaggle" / "cuad"
)


# ---------------------------------------------------------
# JSON LOADER
# ---------------------------------------------------------

def load_json(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# CONTRACT NLI
# ---------------------------------------------------------

def validate_contract_nli():

    print("\n" + "=" * 60)
    print("CONTRACTNLI DATASET")
    print("=" * 60)

    if not CONTRACT_NLI_DIR.exists():

        print(
            f"ERROR: Directory not found:\n"
            f"{CONTRACT_NLI_DIR}"
        )

        return

    json_files = [
        "train.json",
        "dev.json",
        "test.json",
    ]

    for filename in json_files:

        path = CONTRACT_NLI_DIR / filename

        if not path.exists():

            print(f"{filename}: NOT FOUND")
            continue

        data = load_json(path)

        print(f"\n{filename}")

        print(
            "Top-level keys:",
            list(data.keys())
        )

        documents = data.get(
            "documents",
            []
        )

        print(
            "Number of documents:",
            len(documents)
        )

        if documents:

            first_doc = documents[0]

            print(
                "First document keys:",
                list(first_doc.keys())
            )

            text = first_doc.get(
                "text",
                ""
            )

            print(
                "First document text length:",
                len(text)
            )


# ---------------------------------------------------------
# CUAD
# ---------------------------------------------------------

def validate_cuad():

    print("\n" + "=" * 60)
    print("CUAD DATASET")
    print("=" * 60)

    if not CUAD_DIR.exists():

        print(
            f"ERROR: Directory not found:\n"
            f"{CUAD_DIR}"
        )

        return

    json_files = list(
        CUAD_DIR.rglob("*.json")
    )

    if not json_files:

        print(
            "No JSON files found inside CUAD directory."
        )

        return

    print(
        f"Found {len(json_files)} JSON file(s):"
    )

    for path in json_files:

        print(
            f"\n--- {path.relative_to(PROJECT_ROOT)} ---"
        )

        try:

            data = load_json(path)

            print(
                "Top-level type:",
                type(data).__name__
            )

            if isinstance(data, dict):

                print(
                    "Top-level keys:",
                    list(data.keys())[:20]
                )

                # SQuAD-style CUAD
                if "data" in data:

                    articles = data["data"]

                    print(
                        "Number of articles:",
                        len(articles)
                    )

                    if articles:

                        article = articles[0]

                        print(
                            "Article keys:",
                            list(article.keys())
                        )

            elif isinstance(data, list):

                print(
                    "Number of records:",
                    len(data)
                )

        except Exception as error:

            print(
                "ERROR reading file:",
                error
            )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "LexConflict - Dataset Validation"
    )

    print(
        "Project root:",
        PROJECT_ROOT
    )

    validate_contract_nli()
    validate_cuad()

    print("\nValidation completed.")