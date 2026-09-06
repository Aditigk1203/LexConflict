import json
from pathlib import Path


GRAPH_PATH = Path(
    "data/processed/conflict_graph/"
    "dev_conflict_graph.json"
)


def main():

    print("=" * 70)
    print("LEXCONFLICT SAVED GRAPH VALIDATION")
    print("=" * 70)

    assert GRAPH_PATH.exists(), (
        f"Graph file not found: {GRAPH_PATH}"
    )

    with open(
        GRAPH_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        graph = json.load(f)

    assert "nodes" in graph
    assert "edges" in graph

    nodes = graph["nodes"]
    edges = graph["edges"]

    print(
        f"\nNodes: {len(nodes)}"
    )

    print(
        f"Edges: {len(edges)}"
    )

    # -----------------------------------------------------
    # Node validation
    # -----------------------------------------------------

    node_ids = set()

    for node in nodes:

        assert "node_id" in node
        assert "document_id" in node
        assert "clause_id" in node
        assert "text" in node

        node_ids.add(
            node["node_id"]
        )

    # -----------------------------------------------------
    # Edge validation
    # -----------------------------------------------------

    for edge in edges:

        assert "source_id" in edge
        assert "target_id" in edge
        assert "relationship" in edge
        assert "confidence" in edge

        assert (
            edge["source_id"]
            in node_ids
        )

        assert (
            edge["target_id"]
            in node_ids
        )

        assert 0.0 <= float(
            edge["confidence"]
        ) <= 1.0

    # -----------------------------------------------------
    # Conflict validation
    # -----------------------------------------------------

    conflicts = [
        edge
        for edge in edges
        if edge["relationship"] == "conflict"
    ]

    for edge in conflicts:

        assert (
            edge["hybrid_conflict_score"]
            >= 0.71
        )

    print(
        f"\nConflict edges: "
        f"{len(conflicts)}"
    )

    print(
        "\n✓ SAVED GRAPH VALIDATION PASSED"
    )


if __name__ == "__main__":

    main()