from src.graph.graph_propagation import (
    GraphPropagation
)


def main():

    print("=" * 70)
    print("LEXCONFLICT GRAPH PROPAGATION TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # Test graph
    # -----------------------------------------------------

    nodes = [

        {
            "node_id": "A",
            "document_id": "doc1",
            "clause_id": "A",
            "text": "Clause A"
        },

        {
            "node_id": "B",
            "document_id": "doc1",
            "clause_id": "B",
            "text": "Clause B"
        },

        {
            "node_id": "C",
            "document_id": "doc1",
            "clause_id": "C",
            "text": "Clause C"
        },

        {
            "node_id": "D",
            "document_id": "doc1",
            "clause_id": "D",
            "text": "Clause D"
        }
    ]

    edges = [

        {
            "source_id": "A",
            "target_id": "B",
            "relationship": "conflict",
            "hybrid_conflict_score": 0.90,
            "conflict_type":
                "obligation_vs_prohibition"
        },

        {
            "source_id": "B",
            "target_id": "C",
            "relationship": "conflict",
            "hybrid_conflict_score": 0.80,
            "conflict_type":
                "positive_vs_negative"
        },

        {
            "source_id": "C",
            "target_id": "D",
            "relationship": "related",
            "hybrid_conflict_score": 0.20,
            "conflict_type":
                "none"
        }
    ]

    # -----------------------------------------------------
    # Run propagation
    # -----------------------------------------------------

    propagation = GraphPropagation(
        alpha=0.70,
        iterations=3
    )

    results = propagation.propagate(
        nodes,
        edges
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\nPROPAGATION RESULTS")

    for node_id, result in results.items():

        print(
            f"\nNode: {node_id}"
        )

        print(
            "Direct score:",
            round(
                result[
                    "direct_conflict_score"
                ],
                4
            )
        )

        print(
            "Propagated score:",
            round(
                result[
                    "propagated_conflict_score"
                ],
                4
            )
        )

        print(
            "Conflict degree:",
            result[
                "conflict_degree"
            ]
        )

        print(
            "Neighbors:",
            result[
                "conflict_neighbors"
            ]
        )

        print(
            "Hotspot:",
            result[
                "hotspot_level"
            ]
        )

    # -----------------------------------------------------
    # Assertions
    # -----------------------------------------------------

    assert (
        results["A"]["conflict_degree"]
        == 1
    )

    assert (
        results["B"]["conflict_degree"]
        == 2
    )

    assert (
        results["C"]["conflict_degree"]
        == 1
    )

    assert (
        results["D"]["conflict_degree"]
        == 0
    )

    assert (
        results["A"][
            "propagated_conflict_score"
        ] > 0
    )

    assert (
        results["B"][
            "propagated_conflict_score"
        ] > 0
    )

    assert (
        results["D"][
            "propagated_conflict_score"
        ] == 0
    )

    # Related edges must not influence propagation.
    assert (
        "D"
        not in results["C"][
            "conflict_neighbors"
        ]
    )

    print(
        "\n✓ GRAPH PROPAGATION TEST PASSED"
    )


if __name__ == "__main__":

    main()