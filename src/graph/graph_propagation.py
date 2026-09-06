from collections import defaultdict
from typing import Dict, List, Any


class GraphPropagation:

    """
    Propagates conflict influence through the conflict graph.

    The propagation is based only on edges whose relationship
    is explicitly marked as 'conflict'.

    This prevents the large number of 'related' edges from
    overwhelming the conflict signal.
    """

    def __init__(
        self,
        alpha: float = 0.70,
        iterations: int = 3
    ):
        """
        Parameters
        ----------
        alpha:
            Weight retained by the node's original conflict
            signal.

        iterations:
            Number of propagation iterations.
        """

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be between 0 and 1"
            )

        if iterations < 1:
            raise ValueError(
                "iterations must be at least 1"
            )

        self.alpha = alpha
        self.iterations = iterations

    # =====================================================
    # Build conflict adjacency
    # =====================================================

    def build_adjacency(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:

        adjacency = defaultdict(list)

        # Make sure every node exists in the adjacency map.
        for node in nodes:

            node_id = str(
                node["node_id"]
            )

            adjacency[node_id] = []

        # Only conflict edges participate in propagation.
        for edge in edges:

            if edge.get(
                "relationship"
            ) != "conflict":

                continue

            source_id = str(
                edge["source_id"]
            )

            target_id = str(
                edge["target_id"]
            )

            score = float(
                edge.get(
                    "hybrid_conflict_score",
                    edge.get(
                        "confidence",
                        0.0
                    )
                )
            )

            conflict_type = edge.get(
                "conflict_type",
                "unknown"
            )

            edge_info = {
                "neighbor_id": target_id,
                "score": score,
                "conflict_type": conflict_type
            }

            adjacency[source_id].append(
                edge_info
            )

            # Conflict is logically symmetric.
            reverse_edge_info = {
                "neighbor_id": source_id,
                "score": score,
                "conflict_type": conflict_type
            }

            adjacency[target_id].append(
                reverse_edge_info
            )

        return dict(adjacency)

    # =====================================================
    # Calculate initial conflict signal
    # =====================================================

    def calculate_initial_scores(
        self,
        nodes: List[Dict[str, Any]],
        adjacency: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:

        initial_scores = {}

        for node in nodes:

            node_id = str(
                node["node_id"]
            )

            neighbors = adjacency.get(
                node_id,
                []
            )

            if not neighbors:

                initial_scores[node_id] = 0.0

                continue

            # The strongest direct conflict involving
            # this clause becomes its initial signal.
            strongest_score = max(
                neighbor["score"]
                for neighbor in neighbors
            )

            initial_scores[node_id] = float(
                min(
                    strongest_score,
                    1.0
                )
            )

        return initial_scores

    # =====================================================
    # Calculate neighbor influence
    # =====================================================

    def calculate_neighbor_influence(
        self,
        node_id: str,
        current_scores: Dict[str, float],
        adjacency: Dict[str, List[Dict[str, Any]]]
    ) -> float:

        neighbors = adjacency.get(
            node_id,
            []
        )

        if not neighbors:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for neighbor in neighbors:

            neighbor_id = str(
                neighbor["neighbor_id"]
            )

            edge_score = float(
                neighbor["score"]
            )

            neighbor_score = float(
                current_scores.get(
                    neighbor_id,
                    0.0
                )
            )

            weighted_sum += (
                edge_score
                * neighbor_score
            )

            total_weight += edge_score

        if total_weight == 0.0:

            return 0.0

        return float(
            weighted_sum
            / total_weight
        )

    # =====================================================
    # Propagate scores
    # =====================================================

    def propagate(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:

        adjacency = self.build_adjacency(
            nodes,
            edges
        )

        initial_scores = (
            self.calculate_initial_scores(
                nodes,
                adjacency
            )
        )

        current_scores = dict(
            initial_scores
        )

        # -------------------------------------------------
        # Iterative propagation
        # -------------------------------------------------

        for _ in range(
            self.iterations
        ):

            new_scores = {}

            for node in nodes:

                node_id = str(
                    node["node_id"]
                )

                original_score = (
                    initial_scores.get(
                        node_id,
                        0.0
                    )
                )

                neighbor_influence = (
                    self.calculate_neighbor_influence(
                        node_id,
                        current_scores,
                        adjacency
                    )
                )

                propagated_score = (
                    self.alpha
                    * original_score
                    +
                    (1.0 - self.alpha)
                    * neighbor_influence
                )

                new_scores[node_id] = float(
                    min(
                        max(
                            propagated_score,
                            0.0
                        ),
                        1.0
                    )
                )

            current_scores = new_scores

        # -------------------------------------------------
        # Create final node-level results
        # -------------------------------------------------

        results = {}

        for node in nodes:

            node_id = str(
                node["node_id"]
            )

            neighbors = adjacency.get(
                node_id,
                []
            )

            direct_score = float(
                initial_scores.get(
                    node_id,
                    0.0
                )
            )

            propagated_score = float(
                current_scores.get(
                    node_id,
                    0.0
                )
            )

            conflict_degree = len(
                neighbors
            )

            if propagated_score >= 0.80:

                hotspot = "high"

            elif propagated_score >= 0.60:

                hotspot = "medium"

            elif propagated_score >= 0.30:

                hotspot = "low"

            else:

                hotspot = "none"

            results[node_id] = {

                "node_id": node_id,

                "direct_conflict_score":
                    direct_score,

                "propagated_conflict_score":
                    propagated_score,

                "conflict_degree":
                    conflict_degree,

                "conflict_neighbors": [
                    neighbor["neighbor_id"]
                    for neighbor in neighbors
                ],

                "hotspot_level":
                    hotspot
            }

        return results