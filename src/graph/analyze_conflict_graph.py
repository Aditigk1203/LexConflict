import json
from pathlib import Path
from collections import Counter


GRAPH_PATH = Path(
    "data/processed/conflict_graph/"
    "dev_conflict_graph.json"
)


def load_graph():

    with open(
        GRAPH_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def main():

    print("=" * 70)
    print("LEXCONFLICT CONFLICT GRAPH ANALYSIS")
    print("=" * 70)

    graph = load_graph()

    nodes = graph["nodes"]
    edges = graph["edges"]

    conflicts = [
        edge
        for edge in edges
        if edge["relationship"] == "conflict"
    ]

    # -----------------------------------------------------
    # Basic statistics
    # -----------------------------------------------------

    print("\nGRAPH STATISTICS")

    print(
        f"Nodes: {len(nodes)}"
    )

    print(
        f"Edges: {len(edges)}"
    )

    print(
        f"Conflict edges: "
        f"{len(conflicts)}"
    )

    # -----------------------------------------------------
    # Conflict types
    # -----------------------------------------------------

    conflict_types = Counter(
        edge.get(
            "conflict_type",
            "unknown"
        )
        for edge in conflicts
    )

    print("\nCONFLICT TYPES")

    if conflict_types:

        for conflict_type, count in (
            conflict_types.most_common()
        ):

            print(
                f"{conflict_type}: {count}"
            )

    else:

        print(
            "No conflict edges found."
        )

    # -----------------------------------------------------
    # Most problematic clauses
    # -----------------------------------------------------

    conflict_degree = Counter()

    for edge in conflicts:

        source = edge["source_id"]
        target = edge["target_id"]

        conflict_degree[source] += 1
        conflict_degree[target] += 1

    print(
        "\nTOP CONFLICT HOTSPOTS"
    )

    if conflict_degree:

        for (
            clause_id,
            degree
        ) in conflict_degree.most_common(10):

            print(
                f"{clause_id}: "
                f"{degree} conflicts"
            )

    else:

        print(
            "No conflict hotspots found."
        )

    # -----------------------------------------------------
    # Strongest conflicts
    # -----------------------------------------------------

    strongest = sorted(
        conflicts,
        key=lambda edge:
            edge.get(
                "hybrid_conflict_score",
                0.0
            ),
        reverse=True
    )

    print(
        "\nSTRONGEST CONFLICTS"
    )

    for edge in strongest[:10]:

        print(
            "\n"
            f"{edge['source_id']} "
            f"<--> "
            f"{edge['target_id']}"
        )

        print(
            "Type: "
            f"{edge.get('conflict_type')}"
        )

        print(
            "Hybrid score: "
            f"{edge.get('hybrid_conflict_score'):.4f}"
        )

        print(
            "NLI label: "
            f"{edge.get('nli_label')}"
        )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print(
        "\n✓ GRAPH ANALYSIS COMPLETE"
    )


if __name__ == "__main__":

    main()
    