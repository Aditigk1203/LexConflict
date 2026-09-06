from src.models.nli_inference import LegalBERTNLI


MODEL_PATH = "models/lexconflict_legalbert"


pairs = [
    {
        "hypothesis": (
            "The Receiving Party shall not disclose "
            "Confidential Information."
        ),
        "clause_text": (
            "The Receiving Party shall disclose "
            "Confidential Information to third parties."
        )
    },
    {
        "hypothesis": (
            "Payment shall be made within 15 days."
        ),
        "clause_text": (
            "Payment shall be made within 15 days."
        )
    }
]


print("=" * 60)
print("LOADING LEGAL-BERT NLI MODEL")
print("=" * 60)

nli_model = LegalBERTNLI(
    model_path=MODEL_PATH
)


print("\n" + "=" * 60)
print("RUNNING NLI PREDICTION")
print("=" * 60)

predictions = nli_model.predict(
    pairs,
    batch_size=2,
    max_length=120,
    num_workers=0
)


for i, prediction in enumerate(predictions):

    print(f"\nPair {i + 1}")
    print("-" * 40)

    print("Label:", prediction["label"])
    print("Confidence:", prediction["confidence"])
    print("Probabilities:", prediction["probabilities"])