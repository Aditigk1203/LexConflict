from typing import Any, Optional

from .conflict_features import ConflictFeatureExtractor
from .conflict_result import ConflictResult


class ConflictEngine:

    def __init__(
        self,
        conflict_threshold: float = 0.50
    ):
        self.feature_extractor = ConflictFeatureExtractor()
        self.conflict_threshold = conflict_threshold

    # ---------------------------------------------------------
    # Conflict type
    # ---------------------------------------------------------

    def determine_conflict_type(
        self,
        clause_a: Any,
        clause_b: Any,
        features: dict
    ) -> str:

        modality_a = self.feature_extractor.get_value(
            clause_a,
            "modality"
        )

        modality_b = self.feature_extractor.get_value(
            clause_b,
            "modality"
        )

        negation_a = self.feature_extractor.contains_negation(
            self.feature_extractor.get_text(clause_a)
        )

        negation_b = self.feature_extractor.contains_negation(
            self.feature_extractor.get_text(clause_b)
        )

        if (
            modality_a == "obligation"
            and modality_b == "prohibition"
        ):
            return "obligation_vs_prohibition"

        if (
            modality_a == "prohibition"
            and modality_b == "obligation"
        ):
            return "prohibition_vs_obligation"

        if (
            modality_a == "permission"
            and modality_b == "prohibition"
        ):
            return "permission_vs_prohibition"

        if (
            modality_a == "prohibition"
            and modality_b == "permission"
        ):
            return "prohibition_vs_permission"

        if negation_a != negation_b:
            return "positive_vs_negative"

        if features["semantic_similarity"] >= 0.40:
            return "potential_semantic_conflict"

        return "no_clear_conflict"

    # ---------------------------------------------------------
    # Structured conflict score
    # ---------------------------------------------------------

    def calculate_score(
        self,
        features: dict
    ) -> float:

        score = (
            0.35 * features["semantic_similarity"]
            + 0.30 * features["modality_conflict"]
            + 0.20 * features["negation_conflict"]
            + 0.10 * features["same_party"]
            + 0.05 * features["condition_similarity"]
        )

        return float(min(max(score, 0.0), 1.0))

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

    def generate_explanation(
        self,
        conflict_type: str,
        features: dict
    ) -> str:

        reasons = []

        if features["semantic_similarity"] >= 0.40:
            reasons.append(
                "the clauses discuss semantically related content"
            )

        if features["modality_conflict"] > 0:
            reasons.append(
                "their modalities indicate opposing obligations, "
                "permissions, or prohibitions"
            )

        if features["negation_conflict"] > 0:
            reasons.append(
                "one clause contains negation while the other does not"
            )

        if features["same_party"] > 0:
            reasons.append(
                "the clauses refer to the same detected party"
            )

        if features["condition_similarity"] > 0.50:
            reasons.append(
                "their conditions are similar"
            )

        if not reasons:
            return "No strong structured conflict indicators were detected."

        return (
            f"Potential conflict type: {conflict_type}. "
            + "; ".join(reasons)
            + "."
        )

    # ---------------------------------------------------------
    # Analyze pair
    # ---------------------------------------------------------

    def analyze_pair(
        self,
        clause_a: Any,
        clause_b: Any,
        semantic_similarity: Optional[float] = None
    ) -> ConflictResult:

        features = self.feature_extractor.extract_features(
            clause_a,
            clause_b,
            semantic_similarity=semantic_similarity
        )

        score = self.calculate_score(features)

        conflict_type = self.determine_conflict_type(
            clause_a,
            clause_b,
            features
        )

        is_conflict = (
            score >= self.conflict_threshold
            and conflict_type != "no_clear_conflict"
        )

        text_a = self.feature_extractor.get_text(clause_a)
        text_b = self.feature_extractor.get_text(clause_b)

        modality_a = self.feature_extractor.get_value(
            clause_a,
            "modality"
        )

        modality_b = self.feature_extractor.get_value(
            clause_b,
            "modality"
        )

        negation_a = self.feature_extractor.contains_negation(
            text_a
        )

        negation_b = self.feature_extractor.contains_negation(
            text_b
        )

        explanation = self.generate_explanation(
            conflict_type,
            features
        )

        return ConflictResult(
            clause_a_id=self.feature_extractor.get_clause_id(
                clause_a
            ),
            clause_b_id=self.feature_extractor.get_clause_id(
                clause_b
            ),
            clause_a_text=text_a,
            clause_b_text=text_b,
            is_conflict=is_conflict,
            confidence=score,
            conflict_type=conflict_type,
            semantic_similarity=features[
                "semantic_similarity"
            ],
            modality_a=modality_a,
            modality_b=modality_b,
            modality_conflict=features[
                "modality_conflict"
            ],
            negation_a=negation_a,
            negation_b=negation_b,
            negation_conflict=features[
                "negation_conflict"
            ],
            same_party=bool(
                features["same_party"]
            ),
            condition_similarity=features[
                "condition_similarity"
            ],
            explanation=explanation
        )