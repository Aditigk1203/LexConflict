from src.nli.legalbert_inference import LegalBERTInference


MODEL_PATH = r"C:\Users\aditi\Desktop\lexconflict\models\legalbert_contractnli"


def main():

    model = LegalBERTInference(
        MODEL_PATH
    )

    clause_a = (
        "The Supplier shall provide "
        "the report within 30 days."
    )

    clause_b = (
        "The Supplier shall not provide "
        "the report within 30 days."
    )

    result = model.predict(
        clause_a,
        clause_b
    )

    print("\nNLI RESULT")
    print("=" * 60)

    print(
        "Label:",
        result["label"]
    )

    print(
        "Label ID:",
        result["label_id"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    print(
        "Probabilities:",
        result["probabilities"]
    )


if __name__ == "__main__":
    main()