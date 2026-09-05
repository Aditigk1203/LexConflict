from src.retrieval import CandidateRetriever


def main():

    clauses = [

        {
            "document_id": "contract_001",
            "clause_id": "clause_001",
            "text": (
                "The Supplier shall deliver "
                "the goods within 30 days."
            ),
        },

        {
            "document_id": "contract_001",
            "clause_id": "clause_002",
            "text": (
                "The Supplier must provide "
                "the products within thirty days."
            ),
        },

        {
            "document_id": "contract_001",
            "clause_id": "clause_003",
            "text": (
                "The Customer may terminate "
                "the agreement if delivery fails."
            ),
        },

        {
            "document_id": "contract_001",
            "clause_id": "clause_004",
            "text": (
                "The Supplier shall not disclose "
                "confidential information."
            ),
        },

        {
            "document_id": "contract_001",
            "clause_id": "clause_005",
            "text": (
                "Payment shall be made within "
                "15 business days."
            ),
        },

        # Different contract
        {
            "document_id": "contract_002",
            "clause_id": "clause_006",
            "text": (
                "The supplier shall deliver "
                "the goods within 30 days."
            ),
        },
    ]

    # --------------------------------------------------
    # Build retriever
    # --------------------------------------------------

    retriever = CandidateRetriever(
        top_k=3,
        same_document_only=True,
    )

    retriever.fit(
        clauses
    )

    # --------------------------------------------------
    # Query clause
    # --------------------------------------------------

    query_clause = clauses[0]

    results = retriever.retrieve(
        query_clause
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("=" * 70)
    print("LEXCONFLICT CANDIDATE RETRIEVAL TEST")
    print("=" * 70)

    print("\nQUERY CLAUSE:")
    print(
        query_clause["clause_id"]
    )

    print(
        query_clause["text"]
    )

    print("\nRETRIEVED CANDIDATES:")

    for result in results:

        print("\n" + "-" * 70)

        print(
            "Rank:",
            result.rank
        )

        print(
            "Candidate:",
            result.candidate_clause_id
        )

        print(
            "Score:",
            round(
                result.score,
                4
            )
        )

        print(
            "Text:",
            result.candidate_text
        )


if __name__ == "__main__":
    main()