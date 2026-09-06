import json
from pathlib import Path

from src.graph.graph_propagation import (
    GraphPropagation
)


GRAPH_PATH = Path(
    "data/processed/conflict_graph/"
    "dev_conflict_graph.json"
)

OUTPUT_PATH = Path(
    "data/processed/conflict_graph/"
    "dev_graph_propagation.json"
)


def main():

    print("=" * 70)
    print("LEXCONFLICT GRAPH PROPAGATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load graph
    # -----------------------------------------------------

    print(
        "\nLoading graph..."
    )

    with open(
        GRAPH_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        graph = json.load(f)

    nodes = graph[
        "nodes"
    ]

    edges = graph[
        "edges"
    ]

    print(
        f"Nodes: {len(nodes)}"
    )

    print(
        f"Edges: {len(edges)}"
    )

    # -----------------------------------------------------
    # Count conflict edges
    # -----------------------------------------------------

    conflict_edges = [

        edge

        for edge in edges

        if edge.get(
            "relationship"
        ) == "conflict"
    ]

    print(
        f"Conflict edges: "
        f"{len(conflict_edges)}"
    )

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
    # Build output
    # -----------------------------------------------------

    output_nodes = []

    for node in nodes:

        node_id = str(
            node["node_id"]
        )

        propagation_result = results[
            node_id
        ]

        output_nodes.append({

            **node,

            "direct_conflict_score":
                propagation_result[
                    "direct_conflict_score"
                ],

            "propagated_conflict_score":
                propagation_result[
                    "propagated_conflict_score"
                ],

            "conflict_degree":
                propagation_result[
                    "conflict_degree"
                ],

            "conflict_neighbors":
                propagation_result[
                    "conflict_neighbors"
                ],

            "hotspot_level":
                propagation_result[
                    "hotspot_level"
                ]
        })

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    hotspot_counts = {

        "high": 0,
        "medium": 0,
        "low": 0,
        "none": 0
    }

    for result in results.values():

        level = result[
            "hotspot_level"
        ]

        hotspot_counts[level] += 1

    # -----------------------------------------------------
    # Create output
    # -----------------------------------------------------

    output = {

        "phase":
            "Phase 8 - Graph Propagation",

        "parameters": {

            "alpha": 0.70,

            "iterations": 3,

            "propagation_edges":
                "conflict_only"
        },

        "graph_statistics": {

            "number_of_nodes":
                len(nodes),

            "number_of_edges":
                len(edges),

            "number_of_conflict_edges":
                len(conflict_edges)
        },

        "hotspot_statistics":
            hotspot_counts,

        "nodes":
            output_nodes,

        "edges":
            edges
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    # -----------------------------------------------------
    # Print hotspots
    # -----------------------------------------------------

    ranked_nodes = sorted(
        output_nodes,
        key=lambda node:
            node[
                "propagated_conflict_score"
            ],
        reverse=True
    )

    print(
        "\nTOP CONFLICT HOTSPOTS"
    )

    for node in ranked_nodes[:10]:

        if node[
            "propagated_conflict_score"
        ] <= 0:

            continue

        print(
            f"\n{node['node_id']}"
        )

        print(
            "Direct score:",
            round(
                node[
                    "direct_conflict_score"
                ],
                4
            )
        )

        print(
            "Propagated score:",
            round(
                node[
                    "propagated_conflict_score"
                ],
                4
            )
        )

        print(
            "Conflict degree:",
            node[
                "conflict_degree"
            ]
        )

        print(
            "Hotspot:",
            node[
                "hotspot_level"
            ]
        )

    print(
        "\nHOTSPOT COUNTS"
    )

    for level, count in (
        hotspot_counts.items()
    ):

        print(
            f"{level}: {count}"
        )

    print(
        "\nOutput saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\n✓ GRAPH PROPAGATION COMPLETE"
    )


if __name__ == "__main__":
    main()