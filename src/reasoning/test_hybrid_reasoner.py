"""
Integration tests for the LexConflict hybrid reasoning stage.

Tests:
1. ConflictEngine produces structured reasoning.
2. HybridReasoner combines structured reasoning with NLI output.
3. A clear contradiction is classified as a conflict.
4. A clearly unrelated pair is not classified as a conflict.
"""

from src.reasoning.conflict_engine import ConflictEngine
from src.reasoning.hybrid_reasoner import HybridReasoner


def test_clear_conflict():
    """Test an obvious obligation vs prohibition conflict."""

    clause_a = {
        "clause_id": "test_a",
        "text": "Supplier shall provide the report.",
        "party": "supplier",
        "modality": "obligation",
        "condition": None,
    }

    clause_b = {
        "clause_id": "test_b",
        "text": "Supplier shall not provide the report.",
        "party": "supplier",
        "modality": "prohibition",
        "condition": None,
    }

    # Step 1: structured reasoning
    conflict_engine = ConflictEngine()

    structured_result = conflict_engine.analyze_pair(
        clause_a,
        clause_b
    )

    # Step 2: simulate a Legal-BERT NLI result.
    #
    # This is a unit/integration test, so we do not load the
    # large Legal-BERT model here.
    nli_result = {
        "label": "Contradiction",
        "probabilities": [
            0.05,   # Entailment
            0.05,   # NotMentioned
            0.90,   # Contradiction
        ],
    }

    # Step 3: hybrid reasoning
    hybrid_reasoner = HybridReasoner()

    final_result = hybrid_reasoner.analyze(
        nli_result,
        {
            "confidence": structured_result.confidence,
            "conflict_type": structured_result.conflict_type,
        },
    )

    # Verify structured reasoning
    assert structured_result.conflict_type in {
        "obligation_vs_prohibition",
        "prohibition_vs_obligation",
    }

    # Verify hybrid reasoning
    assert final_result["nli_label"] == "Contradiction"

    assert final_result["nli_contradiction_probability"] == 0.90

    assert final_result["structured_conflict_score"] == (
        structured_result.confidence
    )

    # With 0.95 NLI weight:
    # hybrid score should be dominated by the NLI signal.
    expected_score = (
        0.95 * 0.90
        + 0.05 * structured_result.confidence
    )

    assert abs(
        final_result["hybrid_conflict_score"]
        - expected_score
    ) < 1e-9

    assert final_result["is_conflict"] is True

    assert final_result["confidence_level"] in {
        "High",
        "Medium",
        "Low",
    }

    print("PASS: clear conflict integration test")


def test_no_clear_conflict():
    """Test a pair of clauses that should not conflict."""

    clause_a = {
        "clause_id": "test_c",
        "text": "Supplier shall provide the report.",
        "party": "supplier",
        "modality": "obligation",
        "condition": None,
    }

    clause_b = {
        "clause_id": "test_d",
        "text": "Customer shall make payment within 30 days.",
        "party": "customer",
        "modality": "obligation",
        "condition": None,
    }

    # Structured reasoning
    conflict_engine = ConflictEngine()

    structured_result = conflict_engine.analyze_pair(
        clause_a,
        clause_b
    )

    # Simulated Legal-BERT result
    nli_result = {
        "label": "NotMentioned",
        "probabilities": [
            0.05,   # Entailment
            0.90,   # NotMentioned
            0.05,   # Contradiction
        ],
    }

    # Hybrid reasoning
    hybrid_reasoner = HybridReasoner()

    final_result = hybrid_reasoner.analyze(
        nli_result,
        {
            "confidence": structured_result.confidence,
            "conflict_type": structured_result.conflict_type,
        },
    )

    assert final_result["nli_label"] == "NotMentioned"

    assert final_result["nli_contradiction_probability"] == 0.05

    assert final_result["is_conflict"] is False

    print("PASS: no-conflict integration test")


if __name__ == "__main__":
    test_clear_conflict()
    test_no_clear_conflict()

    print()
    print("=" * 60)
    print("ALL HYBRID REASONING TESTS PASSED")
    print("=" * 60)