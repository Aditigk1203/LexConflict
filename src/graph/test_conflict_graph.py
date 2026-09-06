from src.graph.graph_node import GraphNode
from src.graph.graph_edge import GraphEdge
from src.graph.conflict_graph import ConflictGraph


def main():

    print("=" * 70)
    print("LEXCONFLICT CONFLICT GRAPH TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Create graph
    # ---------------------------------------------------------

    graph = ConflictGraph()

    # ---------------------------------------------------------
    # 2. Create nodes
    # ---------------------------------------------------------

    node_a = GraphNode(
        node_id="doc1_clause_001",
        document_id="doc1",
        clause_id="doc1_clause_001",
        text=(
            "The Supplier shall provide "
            "the report within 30 days."
        ),
        party="supplier",
        modality="obligation"
    )

    node_b = GraphNode(
        node_id="doc1_clause_002",
        document_id="doc1",
        clause_id="doc1_clause_002",
        text=(
            "The Supplier shall not provide "
            "the report within 30 days."
        ),
        party="supplier",
        modality="prohibition"
    )

    node_c = GraphNode(
        node_id="doc1_clause_003",
        document_id="doc1",
        clause_id="doc1_clause_003",
        text=(
            "The Supplier shall provide "
            "the invoice within 15 days."
        ),
        party="supplier",
        modality="obligation"
    )

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)

    # ---------------------------------------------------------
    # 3. Add conflict edge
    # ---------------------------------------------------------

    conflict_edge = GraphEdge(
        source_id=node_a.node_id,
        target_id=node_b.node_id,

        relationship="conflict",

        confidence=0.899,

        conflict_type="obligation_vs_prohibition",

        nli_label="Contradiction",

        nli_contradiction_probability=0.90,

        structured_conflict_score=0.88,

        hybrid_conflict_score=0.899,

        explanation=(
            "The clauses contain opposing "
            "obligation and prohibition."
        )
    )

    graph.add_edge(conflict_edge)

    # ---------------------------------------------------------
    # 4. Add related edge
    # ---------------------------------------------------------

    related_edge = GraphEdge(
        source_id=node_a.node_id,
        target_id=node_c.node_id,

        relationship="related",

        confidence=0.30,

        conflict_type="no_clear_conflict",

        nli_label="NotMentioned",

        nli_contradiction_probability=0.05,

        structured_conflict_score=0.10,

        hybrid_conflict_score=0.0525,

        explanation=(
            "The clauses concern the same party "
            "but do not directly conflict."
        )
    )

    graph.add_edge(related_edge)

    # ---------------------------------------------------------
    # 5. Print summary
    # ---------------------------------------------------------

    print("\nGRAPH SUMMARY")

    summary = graph.summary()

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    # ---------------------------------------------------------
    # 6. Test node retrieval
    # ---------------------------------------------------------

    retrieved_node = graph.get_node(
        "doc1_clause_001"
    )

    assert retrieved_node is not None

    assert (
        retrieved_node.clause_id
        == "doc1_clause_001"
    )

    # ---------------------------------------------------------
    # 7. Test conflict retrieval
    # ---------------------------------------------------------

    conflicts = graph.get_conflicts()

    print("\nCONFLICT EDGES")

    for edge in conflicts:

        print(
            f"{edge.source_id} "
            f"--{edge.relationship}--> "
            f"{edge.target_id}"
        )

        print(
            f"Score: "
            f"{edge.hybrid_conflict_score}"
        )

        print(
            f"Type: "
            f"{edge.conflict_type}"
        )

    assert len(conflicts) == 1

    # ---------------------------------------------------------
    # 8. Test node conflicts
    # ---------------------------------------------------------

    node_conflicts = (
        graph.get_conflicts_for_node(
            "doc1_clause_001"
        )
    )

    assert len(node_conflicts) == 1

    # ---------------------------------------------------------
    # 9. Test neighbors
    # ---------------------------------------------------------

    neighbors = graph.get_neighbors(
        "doc1_clause_001"
    )

    print("\nNEIGHBORS")

    for neighbor in neighbors:

        print(neighbor)

    assert "doc1_clause_002" in neighbors
    assert "doc1_clause_003" in neighbors

    # ---------------------------------------------------------
    # 10. Test conflict degree
    # ---------------------------------------------------------

    assert (
        graph.nodes[
            "doc1_clause_001"
        ].conflict_degree == 1
    )

    assert (
        graph.nodes[
            "doc1_clause_002"
        ].conflict_degree == 1
    )

    assert (
        graph.nodes[
            "doc1_clause_003"
        ].conflict_degree == 0
    )

    # ---------------------------------------------------------
    # 11. Final validation
    # ---------------------------------------------------------

    assert graph.number_of_nodes() == 3

    assert graph.number_of_edges() == 2

    assert summary[
        "number_of_conflicts"
    ] == 1

    print(
        "\n✓ CONFLICT GRAPH TEST PASSED"
    )


if __name__ == "__main__":
    main()