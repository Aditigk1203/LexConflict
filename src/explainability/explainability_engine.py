import json
from pathlib import Path
from typing import Dict, List, Optional

from .explanation import ExplanationResult


class ExplainabilityEngine:
    """
    Generates human-readable explanations from
    LexConflict risk and graph information.
    """

    def __init__(
        self,
        risk_scores_path: str,
        graph_path: Optional[str] = None
    ):
        self.risk_scores_path = Path(risk_scores_path)
        self.graph_path = (
            Path(graph_path)
            if graph_path
            else None
        )

        self.risk_scores = []
        self.graph = {}

        self._load_risk_scores()

        if self.graph_path:
            self._load_graph()

    # ---------------------------------------------------------
    # DATA LOADING
    # ---------------------------------------------------------

    def _load_risk_scores(self):
        """
        Load saved risk scores.
        """

        with open(
            self.risk_scores_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.risk_scores = json.load(f)

        if not isinstance(self.risk_scores, list):
            raise ValueError(
                "Risk scores file must contain a list."
            )

    def _load_graph(self):
        """
        Load conflict graph.
        """

        with open(
            self.graph_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.graph = json.load(f)

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    def _find_risk_record(
        self,
        node_id: str
    ) -> Optional[Dict]:
        """
        Find risk information for a particular node.
        """

        for record in self.risk_scores:

            if record.get("node_id") == node_id:
                return record

        return None

    def _get_clause_text(
        self,
        risk_record: Dict
    ) -> str:
        """
        Extract clause text.
        """

        return str(
            risk_record.get(
                "text",
                risk_record.get(
                    "clause_text",
                    ""
                )
            )
        )

    def _get_conflict_types(
        self,
        risk_record: Dict
    ) -> List[str]:
        """
        Extract detected conflict types.
        """

        factors = risk_record.get(
            "risk_factors",
            {}
        )

        conflict_types = factors.get(
            "conflict_types",
            []
        )

        if conflict_types is None:
            return []

        if isinstance(
            conflict_types,
            str
        ):
            return [conflict_types]

        return list(conflict_types)

    # ---------------------------------------------------------
    # FACTOR EXPLANATIONS
    # ---------------------------------------------------------

    def _explain_direct_conflict(
        self,
        score: float
    ) -> str:

        if score >= 0.80:
            return (
                "The clause has a very strong direct "
                "conflict signal."
            )

        if score >= 0.60:
            return (
                "The clause has a strong direct "
                "conflict signal."
            )

        if score >= 0.40:
            return (
                "The clause has a moderate direct "
                "conflict signal."
            )

        if score > 0:
            return (
                "The clause has a relatively weak "
                "direct conflict signal."
            )

        return (
            "No direct conflict signal was detected."
        )

    def _explain_propagation(
        self,
        score: float,
        degree: int
    ) -> str:

        if degree == 0:
            return (
                "The clause is not connected to any "
                "detected conflict edges."
            )

        if score >= 0.80:
            return (
                "Conflict propagation is strong, "
                "indicating that connected conflicting "
                "clauses increase the clause's risk."
            )

        if score >= 0.50:
            return (
                "The clause receives a moderate conflict "
                "signal from connected clauses."
            )

        return (
            "The clause has some conflict connectivity, "
            "but propagated conflict remains limited."
        )

    def _explain_connectivity(
        self,
        degree: int,
        density: float
    ) -> str:

        if degree == 0:
            return (
                "There are no conflict connections."
            )

        return (
            f"The clause is connected to {degree} "
            f"conflict edge(s), with a conflict density "
            f"of {density:.3f}."
        )

    def _explain_severity(
        self,
        severity: float
    ) -> str:

        if severity >= 0.80:
            return (
                "The detected conflict has high severity."
            )

        if severity >= 0.50:
            return (
                "The detected conflict has moderate severity."
            )

        if severity > 0:
            return (
                "The detected conflict has relatively "
                "low severity."
            )

        return (
            "No significant severity contribution was detected."
        )

    # ---------------------------------------------------------
    # RECOMMENDATION
    # ---------------------------------------------------------

    def _generate_recommendation(
        self,
        risk_level: str,
        conflict_degree: int
    ) -> str:

        if risk_level == "critical":

            return (
                "Immediate review recommended. Examine "
                "this clause together with all connected "
                "conflicting clauses and determine which "
                "requirement should prevail."
            )

        if risk_level == "high":

            if conflict_degree > 0:
                return (
                    "Review this clause and its connected "
                    "conflicting clauses. Verify whether "
                    "the conflicting obligations, permissions, "
                    "or prohibitions can coexist."
                )

            return (
                "Review this clause for potential contractual "
                "conflict before finalizing the agreement."
            )

        if risk_level == "medium":

            return (
                "Consider reviewing this clause and checking "
                "its related contractual provisions."
            )

        return (
            "No immediate action is required based on the "
            "current conflict analysis."
        )

    # ---------------------------------------------------------
    # MAIN EXPLANATION
    # ---------------------------------------------------------

    def explain_clause(
        self,
        node_id: str
    ) -> ExplanationResult:

        risk_record = self._find_risk_record(
            node_id
        )

        if risk_record is None:
            raise ValueError(
                f"No risk record found for node: {node_id}"
            )

        clause_text = self._get_clause_text(
            risk_record
        )

        risk_score = float(
            risk_record.get(
                "risk_score",
                0.0
            )
        )

        risk_level = str(
            risk_record.get(
                "risk_level",
                "minimal"
            )
        )

        direct_conflict = float(
            risk_record.get(
                "direct_conflict_score",
                0.0
            )
        )

        propagated_conflict = float(
            risk_record.get(
                "propagated_conflict_score",
                0.0
            )
        )

        conflict_degree = int(
            risk_record.get(
                "conflict_degree",
                0
            )
        )

        conflict_density = float(
            risk_record.get(
                "conflict_density",
                0.0
            )
        )

        severity = float(
            risk_record.get(
                "severity_score",
                0.0
            )
        )

        hotspot_level = str(
            risk_record.get(
                "hotspot_level",
                "none"
            )
        )

        conflict_neighbors = risk_record.get(
            "conflict_neighbors",
            []
        )

        if conflict_neighbors is None:
            conflict_neighbors = []

        conflict_types = self._get_conflict_types(
            risk_record
        )

        # -----------------------------------------------------
        # Build explanation components
        # -----------------------------------------------------

        direct_explanation = (
            self._explain_direct_conflict(
                direct_conflict
            )
        )

        propagation_explanation = (
            self._explain_propagation(
                propagated_conflict,
                conflict_degree
            )
        )

        connectivity_explanation = (
            self._explain_connectivity(
                conflict_degree,
                conflict_density
            )
        )

        severity_explanation = (
            self._explain_severity(
                severity
            )
        )

        # -----------------------------------------------------
        # Conflict type explanation
        # -----------------------------------------------------

        if conflict_types:

            conflict_type_text = (
                "Detected conflict types include: "
                + ", ".join(conflict_types)
                + "."
            )

        else:

            conflict_type_text = (
                "No specific conflict type was recorded."
            )

        # -----------------------------------------------------
        # Full explanation
        # -----------------------------------------------------

        explanation_parts = [

            f"Clause {node_id} has a risk score "
            f"of {risk_score:.3f}, classified as "
            f"{risk_level}.",

            direct_explanation,

            propagation_explanation,

            connectivity_explanation,

            severity_explanation,

            conflict_type_text
        ]

        if conflict_neighbors:

            explanation_parts.append(
                "The clause is connected to the following "
                "conflicting clauses: "
                + ", ".join(conflict_neighbors)
                + "."
            )

        else:

            explanation_parts.append(
                "No conflicting neighboring clauses "
                "were identified."
            )

        explanation = " ".join(
            explanation_parts
        )

        recommendation = (
            self._generate_recommendation(
                risk_level,
                conflict_degree
            )
        )

        contributing_factors = {
            "direct_conflict": direct_conflict,
            "propagated_conflict": propagated_conflict,
            "conflict_degree": conflict_degree,
            "conflict_density": conflict_density,
            "severity": severity,
            "hotspot_level": hotspot_level,
            "conflict_types": conflict_types,
            "conflict_neighbors": conflict_neighbors
        }

        return ExplanationResult(
            node_id=node_id,
            clause_text=clause_text,
            risk_score=risk_score,
            risk_level=risk_level,
            direct_conflict_score=direct_conflict,
            propagated_conflict_score=propagated_conflict,
            conflict_degree=conflict_degree,
            conflict_density=conflict_density,
            severity_score=severity,
            conflict_types=conflict_types,
            conflict_neighbors=conflict_neighbors,
            hotspot_level=hotspot_level,
            contributing_factors=contributing_factors,
            explanation=explanation,
            recommendation=recommendation
        )

    # ---------------------------------------------------------
    # ALL CLAUSES
    # ---------------------------------------------------------

    def explain_all(self) -> List[ExplanationResult]:

        results = []

        for record in self.risk_scores:

            node_id = record.get(
                "node_id"
            )

            if node_id is None:
                continue

            results.append(
                self.explain_clause(
                    node_id
                )
            )

        return results