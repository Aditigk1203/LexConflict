import json
from pathlib import Path


PATH = Path(
    "data/processed/conflict_graph/"
    "dev_graph_propagation.json"
)


def main():

    print("=" * 70)
    print("LEXCONFLICT SAVED GRAPH PROPAGATION VALIDATION")
    print("=" * 70)

    assert PATH.exists(), (
        f"Output not found: {PATH}"
    )

    with open(
        PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    assert (
        data["phase"]
        == "Phase 8 - Graph Propagation"
    )

    assert "nodes" in data
    assert "edges" in data
    assert "parameters" in data
    assert "hotspot_statistics" in data

    nodes = data[
        "nodes"
    ]

    edges = data[
        "edges"
    ]

    assert len(nodes) > 0

    # -----------------------------------------------------
    # Validate node scores
    # -----------------------------------------------------

    for node in nodes:

        direct = float(
            node[
                "direct_conflict_score"
            ]
        )

        propagated = float(
            node[
                "propagated_conflict_score"
            ]
        )

        degree = int(
            node[
                "conflict_degree"
            ]
        )

        assert 0.0 <= direct <= 1.0

        assert 0.0 <= propagated <= 1.0

        assert degree >= 0

        assert node[
            "hotspot_level"
        ] in {
            "none",
            "low",
            "medium",
            "high"
        }

    # -----------------------------------------------------
    # Validate edges
    # -----------------------------------------------------

    conflict_count = 0

    for edge in edges:

        if edge.get(
            "relationship"
        ) == "conflict":

            conflict_count += 1

    assert (
        conflict_count
        == data[
            "graph_statistics"
        ][
            "number_of_conflict_edges"
        ]
    )

    print(
        f"\nNodes validated: "
        f"{len(nodes)}"
    )

    print(
        f"Edges validated: "
        f"{len(edges)}"
    )

    print(
        f"Conflict edges: "
        f"{conflict_count}"
    )

    print(
        "\n✓ SAVED GRAPH PROPAGATION "
        "VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()