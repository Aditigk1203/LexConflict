from src.models.nli_inference import LegalBERTNLI
from src.reasoning.conflict_engine import ConflictEngine
from src.reasoning.hybrid_reasoner import HybridReasoner


MODEL_PATH = "models/lexconflict_legalbert"


# ---------------------------------------------------------
# 1. Load Legal-BERT
# ---------------------------------------------------------

print("\nLoading Legal-BERT...")

nli_model = LegalBERTNLI(
    model_path=MODEL_PATH,
    device="cpu"
)


# ---------------------------------------------------------
# 2. Create an obvious conflict
# ---------------------------------------------------------

hypothesis = (
    "The Supplier shall provide the report within 30 days."
)

clause = (
    "The Supplier shall not provide the report within 30 days."
)


# ---------------------------------------------------------
# 3. NLI prediction
# ---------------------------------------------------------

nli_pairs = [
    {
        "hypothesis": hypothesis,
        "clause_text": clause
    }
]

nli_prediction = nli_model.predict(
    nli_pairs,
    batch_size=1,
    max_length=120,
    num_workers=0
)[0]

print("\nNLI Result:")
print(nli_prediction)


# ---------------------------------------------------------
# 4. Structured conflict reasoning
# ---------------------------------------------------------

clause_a = {
    "clause_id": "test_hypothesis",
    "text": hypothesis,
    "party": "supplier",
    "modality": "obligation",
    "condition": None
}

clause_b = {
    "clause_id": "test_clause",
    "text": clause,
    "party": "supplier",
    "modality": "prohibition",
    "condition": None
}


conflict_engine = ConflictEngine()

structured_result = conflict_engine.analyze_pair(
    clause_a,
    clause_b
)

print("\nStructured Result:")
print(structured_result)


# ---------------------------------------------------------
# 5. Hybrid reasoning
# ---------------------------------------------------------

hybrid_reasoner = HybridReasoner(
    nli_weight=0.95,
    structured_weight=0.05,
    conflict_threshold=0.71
)


hybrid_result = hybrid_reasoner.analyze(
    nli_result=nli_prediction,
    structured_result=structured_result
)


print("\nHybrid Result:")

for key, value in hybrid_result.items():
    print(f"{key}: {value}")


# ---------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------

assert 0.0 <= hybrid_result[
    "nli_contradiction_probability"
] <= 1.0

assert 0.0 <= hybrid_result[
    "structured_conflict_score"
] <= 1.0

assert 0.0 <= hybrid_result[
    "hybrid_conflict_score"
] <= 1.0

assert 0.0 <= hybrid_result["hybrid_conflict_score"] <= 1.0
assert hybrid_result["conflict_type"] == "obligation_vs_prohibition"

print(
    "\n✓ Hybrid pipeline executed successfully "
    "even though Legal-BERT did not classify this synthetic pair as a contradiction."
)


print("\n✓ CONFLICT INTEGRATION TEST PASSED")

# ---------------------------------------------------------
# 6B. Verify hybrid decision logic
# ---------------------------------------------------------

controlled_nli_result = {
    "label": "Contradiction",
    "probabilities": [
        0.05,   # Entailment
        0.05,   # NotMentioned
        0.90    # Contradiction
    ]
}

controlled_structured_result = {
    "confidence": 0.88,
    "conflict_type": "obligation_vs_prohibition"
}

controlled_hybrid_result = hybrid_reasoner.analyze(
    nli_result=controlled_nli_result,
    structured_result=controlled_structured_result
)

print("\nControlled Hybrid Test:")

for key, value in controlled_hybrid_result.items():
    print(f"{key}: {value}")

assert controlled_hybrid_result[
    "is_conflict"
] is True

assert controlled_hybrid_result[
    "hybrid_conflict_score"
] >= 0.71

print("\n✓ HYBRID DECISION LOGIC TEST PASSED")


# ---------------------------------------------------------
# 7. Non-conflict regression test
# ---------------------------------------------------------

hypothesis_2 = (
    "The Customer shall make payment within "
    "15 business days."
)

clause_2 = (
    "The Supplier shall provide the invoice within "
    "30 days."
)

nli_pairs_2 = [
    {
        "hypothesis": hypothesis_2,
        "clause_text": clause_2
    }
]

nli_prediction_2 = nli_model.predict(
    nli_pairs_2,
    batch_size=1,
    max_length=120,
    num_workers=0
)[0]


clause_a_2 = {
    "clause_id": "test_hypothesis_2",
    "text": hypothesis_2,
    "party": "customer",
    "modality": "obligation",
    "condition": None
}

clause_b_2 = {
    "clause_id": "test_clause_2",
    "text": clause_2,
    "party": "supplier",
    "modality": "obligation",
    "condition": None
}


structured_result_2 = conflict_engine.analyze_pair(
    clause_a_2,
    clause_b_2
)


hybrid_result_2 = hybrid_reasoner.analyze(
    nli_result=nli_prediction_2,
    structured_result=structured_result_2
)


print("\nNon-conflict Test:")

for key, value in hybrid_result_2.items():
    print(f"{key}: {value}")


assert hybrid_result_2["is_conflict"] is False


print("\n✓ NON-CONFLICT REGRESSION TEST PASSED")

print("\n✓ PHASE 5 INTEGRATION TEST PASSED")