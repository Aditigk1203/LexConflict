from typing import Dict, List, Any

from .risk_result import RiskResult


class RiskEngine:
    """
    Converts graph conflict information into clause-level legal risk.

    The engine combines:

    1. Direct conflict score
    2. Propagated conflict score
    3. Conflict connectivity
    4. Conflict-type severity

    The weights are engineering starting points and are not
    presented as statistically validated final weights.
    """

    def __init__(
        self,
        propagated_weight: float = 0.50,
        direct_weight: float = 0.30,
        connectivity_weight: float = 0.20,
    ):

        total = (
            propagated_weight
            + direct_weight
            + connectivity_weight
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Risk weights must sum to 1.0"
            )

        self.propagated_weight = propagated_weight
        self.direct_weight = direct_weight
        self.connectivity_weight = connectivity_weight

    # ---------------------------------------------------------
    # Conflict-type severity
    # ---------------------------------------------------------

    def conflict_type_severity(
        self,
        conflict_types: List[str]
    ) -> float:

        if not conflict_types:
            return 0.0

        severity_values = []

        for conflict_type in conflict_types:

            if conflict_type in {
                "obligation_vs_prohibition",
                "prohibition_vs_obligation",
            }:
                severity_values.append(1.00)

            elif conflict_type == "positive_vs_negative":
                severity_values.append(0.90)

            elif conflict_type in {
                "permission_vs_prohibition",
                "prohibition_vs_permission",
            }:
                severity_values.append(0.85)

            elif conflict_type == "potential_semantic_conflict":
                severity_values.append(0.70)

            elif conflict_type == "no_clear_conflict":
                severity_values.append(0.30)

            else:
                severity_values.append(0.50)

        return max(severity_values)

    # ---------------------------------------------------------
    # Risk level
    # ---------------------------------------------------------

    def classify_risk(
        self,
        risk_score: float
    ) -> str:

        if risk_score >= 0.80:
            return "critical"

        if risk_score >= 0.60:
            return "high"

        if risk_score >= 0.35:
            return "medium"

        if risk_score >= 0.15:
            return "low"

        return "minimal"

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

    def generate_explanation(
        self,
        node_id: str,
        direct_score: float,
        propagated_score: float,
        conflict_degree: int,
        conflict_density: float,
        severity_score: float,
        risk_score: float,
        risk_level: str,
        conflict_types: List[str],
    ) -> str:

        if conflict_degree == 0:
            return (
                f"Clause {node_id} has no detected conflict "
                f"connections in the current graph. "
                f"Its risk is therefore {risk_level} "
                f"with a score of {risk_score:.3f}."
            )

        type_text = (
            ", ".join(conflict_types)
            if conflict_types
            else "unspecified conflict"
        )

        return (
            f"Clause {node_id} has a direct conflict score of "
            f"{direct_score:.3f} and a propagated conflict score "
            f"of {propagated_score:.3f}. It is connected to "
            f"{conflict_degree} conflict edge(s), giving a "
            f"connectivity value of {conflict_density:.3f}. "
            f"The strongest detected conflict type is "
            f"{type_text}. These factors produce a final risk "
            f"score of {risk_score:.3f}, classified as "
            f"{risk_level}."
        )

    # ---------------------------------------------------------
    # Calculate risk for one node
    # ---------------------------------------------------------

    def calculate_node_risk(
        self,
        node_id: str,
        direct_score: float,
        propagated_score: float,
        conflict_degree: int,
        max_conflict_degree: int,
        conflict_types: List[str],
    ) -> RiskResult:

        direct_score = max(
            0.0,
            min(1.0, float(direct_score))
        )

        propagated_score = max(
            0.0,
            min(1.0, float(propagated_score))
        )

        if max_conflict_degree > 0:
            conflict_density = (
                float(conflict_degree)
                / float(max_conflict_degree)
            )
        else:
            conflict_density = 0.0

        conflict_density = max(
            0.0,
            min(1.0, conflict_density)
        )

        severity_score = self.conflict_type_severity(
            conflict_types
        )

        base_risk = (
            self.propagated_weight * propagated_score
            + self.direct_weight * direct_score
            + self.connectivity_weight * conflict_density
        )

        # Severity acts as a modifier rather than replacing
        # the graph-derived evidence.
        if conflict_degree > 0:
            risk_score = (
                base_risk
                * (0.70 + 0.30 * severity_score)
            )
        else:
            risk_score = (
                base_risk
                * 0.50
            )

        risk_score = max(
            0.0,
            min(1.0, risk_score)
        )

        risk_level = self.classify_risk(
            risk_score
        )

        explanation = self.generate_explanation(
            node_id=node_id,
            direct_score=direct_score,
            propagated_score=propagated_score,
            conflict_degree=conflict_degree,
            conflict_density=conflict_density,
            severity_score=severity_score,
            risk_score=risk_score,
            risk_level=risk_level,
            conflict_types=conflict_types,
        )

        return RiskResult(
            node_id=node_id,
            direct_conflict_score=direct_score,
            propagated_conflict_score=propagated_score,
            conflict_degree=int(conflict_degree),
            conflict_density=conflict_density,
            severity_score=severity_score,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_factors={
                "conflict_types": conflict_types,
                "propagated_weight": self.propagated_weight,
                "direct_weight": self.direct_weight,
                "connectivity_weight": self.connectivity_weight,
            },
            explanation=explanation,
        )

    # ---------------------------------------------------------
    # Calculate risk for complete graph
    # ---------------------------------------------------------

    def calculate_graph_risk(
        self,
        propagation_data: Dict[str, Any],
        graph_data: Dict[str, Any],
    ) -> List[RiskResult]:

        nodes = propagation_data.get(
            "nodes",
            propagation_data.get("results", [])
        )

        edges = graph_data.get(
            "edges",
            []
        )

        if not nodes:
            raise ValueError(
                "No node propagation data found."
            )

        # -----------------------------------------------------
        # Build conflict-type lookup
        # -----------------------------------------------------

        conflict_types_by_node = {}

        conflict_degree_by_node = {}

        for node in nodes:

            node_id = str(
                node.get(
                    "node_id",
                    node.get("id", "")
                )
            )

            conflict_degree = int(
                node.get(
                    "conflict_degree",
                    0
                )
            )

            conflict_degree_by_node[node_id] = (
                conflict_degree
            )

            conflict_types_by_node[node_id] = []

        # -----------------------------------------------------
        # Read conflict edges
        # -----------------------------------------------------

        for edge in edges:

            relationship = edge.get(
                "relationship",
                edge.get("relation", "")
            )

            if relationship != "conflict":
                continue

            source = str(
                edge.get(
                    "source_id",
                    edge.get("source", "")
                )
            )

            target = str(
                edge.get(
                    "target_id",
                    edge.get("target", "")
                )
            )

            conflict_type = edge.get(
                "conflict_type",
                "unknown"
            )

            if source not in conflict_types_by_node:
                conflict_types_by_node[source] = []

            if target not in conflict_types_by_node:
                conflict_types_by_node[target] = []

            conflict_types_by_node[source].append(
                conflict_type
            )

            conflict_types_by_node[target].append(
                conflict_type
            )

        # -----------------------------------------------------
        # Maximum degree for normalization
        # -----------------------------------------------------

        max_conflict_degree = max(
            conflict_degree_by_node.values(),
            default=0
        )

        # -----------------------------------------------------
        # Calculate node risks
        # -----------------------------------------------------

        results = []

        for node in nodes:

            node_id = str(
                node.get(
                    "node_id",
                    node.get("id", "")
                )
            )

            direct_score = float(
                node.get(
                    "direct_score",
                    node.get(
                        "direct_conflict_score",
                        0.0
                    )
                )
            )

            propagated_score = float(
                node.get(
                    "propagated_score",
                    node.get(
                        "propagated_conflict_score",
                        direct_score
                    )
                )
            )

            conflict_degree = int(
                node.get(
                    "conflict_degree",
                    0
                )
            )

            conflict_types = conflict_types_by_node.get(
                node_id,
                []
            )

            result = self.calculate_node_risk(
                node_id=node_id,
                direct_score=direct_score,
                propagated_score=propagated_score,
                conflict_degree=conflict_degree,
                max_conflict_degree=max_conflict_degree,
                conflict_types=conflict_types,
            )

            results.append(result)

        return results

    # ---------------------------------------------------------
    # Contract/document summary
    # ---------------------------------------------------------

    def summarize_risk(
        self,
        risk_results: List[RiskResult]
    ) -> Dict[str, Any]:

        if not risk_results:
            return {
                "number_of_nodes": 0,
                "average_risk": 0.0,
                "maximum_risk": 0.0,
                "risk_level_counts": {},
                "high_risk_nodes": [],
            }

        risk_values = [
            result.risk_score
            for result in risk_results
        ]

        level_counts = {}

        for result in risk_results:

            level = result.risk_level

            level_counts[level] = (
                level_counts.get(level, 0) + 1
            )

        ranked = sorted(
            risk_results,
            key=lambda x: x.risk_score,
            reverse=True
        )

        high_risk_nodes = []

        for result in ranked[:10]:

            high_risk_nodes.append({
                "node_id": result.node_id,
                "risk_score": round(
                    result.risk_score,
                    4
                ),
                "risk_level": result.risk_level,
                "conflict_degree": result.conflict_degree,
            })

        average_risk = sum(
            risk_values
        ) / len(risk_values)

        maximum_risk = max(
            risk_values
        )

        overall_level = self.classify_risk(
            maximum_risk
        )

        return {
            "number_of_nodes": len(risk_results),
            "average_risk": round(
                average_risk,
                4
            ),
            "maximum_risk": round(
                maximum_risk,
                4
            ),
            "overall_risk_level": overall_level,
            "risk_level_counts": level_counts,
            "high_risk_nodes": high_risk_nodes,
        }