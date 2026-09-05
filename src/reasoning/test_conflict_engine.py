from src.reasoning import ConflictEngine


def main():

    # ==================================================
    # TEST 1 — Clear conflict
    # ==================================================

    clause_a = {
        "document_id": "contract_001",
        "clause_id": "clause_001",
        "text": (
            "The Supplier shall provide "
            "the report within 30 days."
        ),
        "party": "supplier",
        "modality": "obligation",
        "condition": None,
        "proposition": (
            "The Supplier provide "
            "the report within 30 days."
        ),
    }

    clause_b = {
        "document_id": "contract_001",
        "clause_id": "clause_002",
        "text": (
            "The Supplier shall not provide "
            "the report within 30 days."
        ),
        "party": "supplier",
        "modality": "prohibition",
        "condition": None,
        "proposition": (
            "The Supplier provide "
            "the report within 30 days."
        ),
    }

    # ==================================================
    # TEST 2 — Clearly unrelated
    # ==================================================

    clause_c = {
        "document_id": "contract_001",
        "clause_id": "clause_003",
        "text": (
            "Payment shall be made "
            "within 15 business days."
        ),
        "party": "customer",
        "modality": "obligation",
        "condition": None,
        "proposition": (
            "Payment made within "
            "15 business days."
        ),
    }

    # ==================================================
    # Create engine
    # ==================================================

    engine = ConflictEngine(
        conflict_threshold=0.50
    )

    # ==================================================
    # Analyze A vs B
    # ==================================================

    result_1 = engine.analyze_pair(
        clause_a,
        clause_b,
    )

    print("=" * 70)
    print("TEST 1 — EXPECTED CONFLICT")
    print("=" * 70)

    print(
        "\nConflict:",
        result_1.is_conflict
    )

    print(
        "Confidence:",
        round(
            result_1.confidence,
            4
        )
    )

    print(
        "Type:",
        result_1.conflict_type
    )

    print(
        "Explanation:",
        result_1.explanation
    )

    # ==================================================
    # Analyze A vs C
    # ==================================================

    result_2 = engine.analyze_pair(
        clause_a,
        clause_c,
    )

    print("\n" + "=" * 70)
    print("TEST 2 — EXPECTED NON-CONFLICT")
    print("=" * 70)

    print(
        "\nConflict:",
        result_2.is_conflict
    )

    print(
        "Confidence:",
        round(
            result_2.confidence,
            4
        )
    )

    print(
        "Type:",
        result_2.conflict_type
    )

    print(
        "Explanation:",
        result_2.explanation
    )


if __name__ == "__main__":
    main()