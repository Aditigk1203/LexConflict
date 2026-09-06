from typing import Dict, List, Optional

from src.graph.graph_node import GraphNode
from src.graph.graph_edge import GraphEdge


class ConflictGraph:

    def __init__(self):

        self.nodes: Dict[str, GraphNode] = {}

        self.edges: List[GraphEdge] = []

        self.adjacency: Dict[str, List[GraphEdge]] = {}

    # ---------------------------------------------------------
    # Add node
    # ---------------------------------------------------------

    def add_node(self, node: GraphNode):

        if node.node_id not in self.nodes:

            self.nodes[node.node_id] = node

            self.adjacency[node.node_id] = []

    # ---------------------------------------------------------
    # Add edge
    # ---------------------------------------------------------

    def add_edge(self, edge: GraphEdge):

        if edge.source_id not in self.nodes:
            raise ValueError(
                f"Unknown source node: {edge.source_id}"
            )

        if edge.target_id not in self.nodes:
            raise ValueError(
                f"Unknown target node: {edge.target_id}"
            )

        self.edges.append(edge)

        self.adjacency[
            edge.source_id
        ].append(edge)

        # Update conflict degree
        if edge.relationship == "conflict":

            self.nodes[
                edge.source_id
            ].conflict_degree += 1

            self.nodes[
                edge.target_id
            ].conflict_degree += 1

    # ---------------------------------------------------------
    # Get node
    # ---------------------------------------------------------

    def get_node(
        self,
        node_id: str
    ) -> Optional[GraphNode]:

        return self.nodes.get(node_id)

    # ---------------------------------------------------------
    # Get all nodes
    # ---------------------------------------------------------

    def get_nodes(self) -> List[GraphNode]:

        return list(
            self.nodes.values()
        )

    # ---------------------------------------------------------
    # Get all edges
    # ---------------------------------------------------------

    def get_edges(self) -> List[GraphEdge]:

        return list(self.edges)

    # ---------------------------------------------------------
    # Get conflicts
    # ---------------------------------------------------------

    def get_conflicts(self) -> List[GraphEdge]:

        return [
            edge
            for edge in self.edges
            if edge.relationship == "conflict"
        ]

    # ---------------------------------------------------------
    # Get conflicts for a clause
    # ---------------------------------------------------------

    def get_conflicts_for_node(
        self,
        node_id: str
    ) -> List[GraphEdge]:

        results = []

        for edge in self.edges:

            if (
                edge.relationship == "conflict"
                and (
                    edge.source_id == node_id
                    or edge.target_id == node_id
                )
            ):

                results.append(edge)

        return results

    # ---------------------------------------------------------
    # Get neighbors
    # ---------------------------------------------------------

    def get_neighbors(
        self,
        node_id: str
    ) -> List[str]:

        neighbors = []

        for edge in self.edges:

            if edge.source_id == node_id:

                neighbors.append(
                    edge.target_id
                )

            elif edge.target_id == node_id:

                neighbors.append(
                    edge.source_id
                )

        return list(
            dict.fromkeys(neighbors)
        )

    # ---------------------------------------------------------
    # Number of nodes
    # ---------------------------------------------------------

    def number_of_nodes(self) -> int:

        return len(self.nodes)

    # ---------------------------------------------------------
    # Number of edges
    # ---------------------------------------------------------

    def number_of_edges(self) -> int:

        return len(self.edges)
    
        # ---------------------------------------------------------
    # Add hybrid conflict result
    # ---------------------------------------------------------

    def add_hybrid_result(
        self,
        source_id: str,
        target_id: str,
        hybrid_result: Dict
    ):

        relationship = (
            "conflict"
            if hybrid_result.get("is_conflict", False)
            else "related"
        )

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,

            relationship=relationship,

            confidence=float(
                hybrid_result.get(
                    "hybrid_conflict_score",
                    0.0
                )
            ),

            conflict_type=hybrid_result.get(
                "conflict_type"
            ),

            nli_label=hybrid_result.get(
                "nli_label"
            ),

            nli_contradiction_probability=float(
                hybrid_result.get(
                    "nli_contradiction_probability",
                    0.0
                )
            ),

            structured_conflict_score=float(
                hybrid_result.get(
                    "structured_conflict_score",
                    0.0
                )
            ),

            hybrid_conflict_score=float(
                hybrid_result.get(
                    "hybrid_conflict_score",
                    0.0
                )
            ),

            explanation=hybrid_result.get(
                "explanation"
            )
        )

        self.add_edge(edge)

        return edge

    # ---------------------------------------------------------
    # Graph summary
    # ---------------------------------------------------------

    def summary(self) -> Dict:

        conflict_edges = self.get_conflicts()

        return {
            "number_of_nodes":
                self.number_of_nodes(),

            "number_of_edges":
                self.number_of_edges(),

            "number_of_conflicts":
                len(conflict_edges)
        }