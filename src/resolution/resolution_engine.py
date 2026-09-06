from .severity import calculate_severity
from .recommendation import generate_recommendation
from .resolution_result import ResolutionResult


class ConflictResolutionEngine:

    def resolve(self, graph_edge):

        severity = calculate_severity(
            graph_edge.hybrid_conflict_score
        )

        recommendation = generate_recommendation(
            graph_edge.conflict_type
        )

        explanation = graph_edge.explanation

        return ResolutionResult(
            severity=severity,
            recommendation=recommendation,
            explanation=explanation,
            suggested_revision=recommendation
        )