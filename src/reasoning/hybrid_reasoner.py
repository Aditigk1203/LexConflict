from typing import Dict, Any


class HybridReasoner:

    def __init__(
        self,
        nli_weight: float = 0.95,
        structured_weight: float = 0.05,
        conflict_threshold: float = 0.71
    ):
        """
        Combines Legal-BERT NLI contradiction probability
        with the structured conflict score.

        These weights are baseline heuristic weights.
        They are NOT final optimized weights.
        """

        self.nli_weight = nli_weight
        self.structured_weight = structured_weight
        self.conflict_threshold = conflict_threshold

    # ---------------------------------------------------------
    # Hybrid score
    # ---------------------------------------------------------

    def calculate_hybrid_score(
        self,
        nli_contradiction_probability: float,
        structured_conflict_score: float
    ) -> float:

        score = (
            self.nli_weight *
            nli_contradiction_probability
            +
            self.structured_weight *
            structured_conflict_score
        )

        return float(
            min(max(score, 0.0), 1.0)
        )

    # ---------------------------------------------------------
    # Final decision
    # ---------------------------------------------------------

    def predict_conflict(
        self,
        nli_contradiction_probability: float,
        structured_conflict_score: float
    ) -> bool:

        hybrid_score = self.calculate_hybrid_score(
            nli_contradiction_probability,
            structured_conflict_score
        )

        return hybrid_score >= self.conflict_threshold

    # ---------------------------------------------------------
    # Confidence level
    # ---------------------------------------------------------

    def confidence_level(
        self,
        hybrid_score: float
    ) -> str:

        if hybrid_score >= 0.75:
            return "High"

        if hybrid_score >= 0.50:
            return "Medium"

        return "Low"

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

    def generate_explanation(
        self,
        nli_label: str,
        nli_contradiction_probability: float,
        structured_conflict_score: float,
        structured_conflict_type: str
    ) -> str:

        reasons = []

        if nli_label == "Contradiction":
            reasons.append(
                f"Legal-BERT detected contradiction "
                f"with probability "
                f"{nli_contradiction_probability:.2f}"
            )

        elif nli_label == "Entailment":
            reasons.append(
                f"Legal-BERT detected entailment "
                f"with probability "
                f"{1 - nli_contradiction_probability:.2f}"
            )

        else:
            reasons.append(
                "Legal-BERT did not detect a strong contradiction"
            )

        if structured_conflict_score >= 0.50:
            reasons.append(
                f"structured reasoning detected conflict indicators "
                f"(score={structured_conflict_score:.2f})"
            )
        else:
            reasons.append(
                f"structured reasoning found limited conflict indicators "
                f"(score={structured_conflict_score:.2f})"
            )

        if structured_conflict_type != "no_clear_conflict":
            reasons.append(
                f"conflict pattern: {structured_conflict_type}"
            )

        return "Hybrid reasoning: " + "; ".join(reasons) + "."

    # ---------------------------------------------------------
    # Full analysis
    # ---------------------------------------------------------

    def analyze(
        self,
        nli_result: Dict[str, Any],
        structured_result: Any
    ) -> Dict[str, Any]:

    # ---------------------------------------------------------
    # NLI result
    # ---------------------------------------------------------

        nli_label = nli_result["label"]

        probabilities = nli_result.get(
            "probabilities",
            []
        )

    # Label mapping:
    # 0 = Entailment
    # 1 = NotMentioned
    # 2 = Contradiction

        if len(probabilities) >= 3:
            contradiction_probability = float(
             probabilities[2]
            )
        else:
            contradiction_probability = (
                1.0
                if nli_label == "Contradiction"
                else 0.0
            )

    # ---------------------------------------------------------
    # Structured result
    # Supports both:
    #   1. ConflictResult dataclass
    #   2. Dictionary
    # ---------------------------------------------------------

        if isinstance(structured_result, dict):

            structured_score = float(
                structured_result.get(
                    "confidence",
                    0.0
                )
            )

            conflict_type = structured_result.get(
                "conflict_type",
                "unknown"
            )

        else:

            structured_score = float(
                getattr(
                    structured_result,
                    "confidence",
                    0.0
                )
            )

            conflict_type = getattr(
                structured_result,
                "conflict_type",
                "unknown"
            )

    # ---------------------------------------------------------
    # Hybrid score
    # ---------------------------------------------------------

        hybrid_score = self.calculate_hybrid_score(
            contradiction_probability,
            structured_score
        )

    # ---------------------------------------------------------
    # Final conflict decision
    # ---------------------------------------------------------

        is_conflict = (
            hybrid_score >= self.conflict_threshold
        )

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

        level = self.confidence_level(
            hybrid_score
        )

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

        explanation = self.generate_explanation(
            nli_label,
            contradiction_probability,
            structured_score,
            conflict_type
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

        return {
            "nli_label": nli_label,

            "nli_contradiction_probability":
                contradiction_probability,

            "structured_conflict_score":
                structured_score,

            "hybrid_conflict_score":
                hybrid_score,

            "is_conflict":
                is_conflict,

            "confidence_level":
                level,

            "conflict_type":
                conflict_type,

            "explanation":
                explanation
        }