from typing import Dict, Any


class HybridReasoner:

    def __init__(
        self,
        nli_weight: float = 0.95,
        structured_weight: float = 0.05,
        conflict_threshold: float = 0.71,
        temporal_weight: float = 0.0
    ):
        """
        Combines:

        1. Legal-BERT contradiction probability
        2. Structured conflict score
        3. Temporal/numeric conflict score

        temporal_weight defaults to 0.0 so that the original
        ContractNLI hybrid tests continue to behave exactly
        as before.
        """

        self.nli_weight = nli_weight

        self.structured_weight = (
            structured_weight
        )

        self.temporal_weight = (
            temporal_weight
        )

        self.conflict_threshold = (
            conflict_threshold
        )


        total_weight = (
            self.nli_weight
            + self.structured_weight
            + self.temporal_weight
        )


        if abs(
            total_weight - 1.0
        ) > 1e-6:

            raise ValueError(
                "Hybrid weights must sum to 1.0"
            )


    # ========================================================
    # HYBRID SCORE
    # ========================================================

    def calculate_hybrid_score(
        self,
        nli_contradiction_probability: float,
        structured_conflict_score: float,
        temporal_conflict_score: float = 0.0
    ) -> float:

        score = (

            self.nli_weight
            * nli_contradiction_probability

            +

            self.structured_weight
            * structured_conflict_score

            +

            self.temporal_weight
            * temporal_conflict_score
        )


        return float(
            min(
                max(
                    score,
                    0.0
                ),
                1.0
            )
        )


    # ========================================================
    # FINAL DECISION
    # ========================================================

    def predict_conflict(
        self,
        nli_contradiction_probability: float,
        structured_conflict_score: float,
        temporal_conflict_score: float = 0.0
    ) -> bool:

        hybrid_score = (
            self.calculate_hybrid_score(

                nli_contradiction_probability,

                structured_conflict_score,

                temporal_conflict_score
            )
        )


        return (
            hybrid_score
            >=
            self.conflict_threshold
        )


    # ========================================================
    # CONFIDENCE LEVEL
    # ========================================================

    def confidence_level(
        self,
        hybrid_score: float
    ) -> str:

        if hybrid_score >= 0.75:

            return "High"


        if hybrid_score >= 0.50:

            return "Medium"


        return "Low"


    # ========================================================
    # EXPLANATION
    # ========================================================

    def generate_explanation(
        self,
        nli_label: str,
        nli_contradiction_probability: float,
        structured_conflict_score: float,
        structured_conflict_type: str,
        temporal_conflict_score: float = 0.0,
        temporal_conflict_type: str = (
            "no_temporal_conflict"
        )
    ) -> str:

        reasons = []


        # ----------------------------------------------------
        # NLI
        # ----------------------------------------------------

        if nli_label == "Contradiction":

            reasons.append(

                "Legal-BERT detected contradiction "
                f"with probability "
                f"{nli_contradiction_probability:.2f}"
            )


        elif nli_label == "Entailment":

            reasons.append(

                "Legal-BERT detected entailment "
                f"with probability "
                f"{1 - nli_contradiction_probability:.2f}"
            )


        else:

            reasons.append(

                "Legal-BERT did not detect "
                "a strong contradiction"
            )


        # ----------------------------------------------------
        # Structured reasoning
        # ----------------------------------------------------

        if structured_conflict_score >= 0.50:

            reasons.append(

                "structured reasoning detected "
                "conflict indicators "
                f"(score="
                f"{structured_conflict_score:.2f})"
            )

        else:

            reasons.append(

                "structured reasoning found "
                "limited conflict indicators "
                f"(score="
                f"{structured_conflict_score:.2f})"
            )


        if (
            structured_conflict_type
            !=
            "no_clear_conflict"
        ):

            reasons.append(

                f"conflict pattern: "
                f"{structured_conflict_type}"
            )


        # ----------------------------------------------------
        # Temporal reasoning
        # ----------------------------------------------------

        if temporal_conflict_score > 0:

            reasons.append(

                "temporal analysis detected "
                f"{temporal_conflict_type} "
                f"(score="
                f"{temporal_conflict_score:.2f})"
            )

        

        return (
            "Hybrid reasoning: "
            +
            "; ".join(reasons)
            +
            "."
        )


    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    def analyze(
        self,
        nli_result: Dict[str, Any],
        structured_result: Any,
        temporal_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # NLI
        # ----------------------------------------------------

        nli_label = nli_result[
            "label"
        ]


        probabilities = (
            nli_result.get(
                "probabilities",
                []
            )
        )


        if len(
            probabilities
        ) >= 3:

            contradiction_probability = float(
                probabilities[2]
            )

        else:

            contradiction_probability = (

                1.0

                if nli_label
                ==
                "Contradiction"

                else

                0.0
            )


        # ----------------------------------------------------
        # Structured reasoning
        # ----------------------------------------------------

        if isinstance(
            structured_result,
            dict
        ):

            structured_score = float(

                structured_result.get(

                    "confidence",

                    0.0
                )
            )


            structured_type = (
                structured_result.get(

                    "conflict_type",

                    "unknown"
                )
            )


        else:

            structured_score = float(

                getattr(

                    structured_result,

                    "confidence",

                    0.0
                )
            )


            structured_type = (

                getattr(

                    structured_result,

                    "conflict_type",

                    "unknown"
                )
            )


        # ----------------------------------------------------
        # Temporal reasoning
        # ----------------------------------------------------

        temporal_result = (
            temporal_result
            or {}
        )


        temporal_score = float(

            temporal_result.get(

                "temporal_conflict_score",

                0.0
            )
        )


        temporal_type = (
            temporal_result.get(

                "conflict_type",

                "no_temporal_conflict"
            )
        )


        # ----------------------------------------------------
        # Final conflict type
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Final conflict type
        # ----------------------------------------------------

        # Temporal reasoning has priority when it has
        # identified a real temporal conflict.
        if (
            temporal_score > 0
            and temporal_type != "no_temporal_conflict"
        ):

            final_conflict_type = temporal_type

        # Otherwise use the structured reasoning result.
        elif (
            structured_type
            and structured_type != "unknown"
        ):

            final_conflict_type = structured_type

        else:
        
            final_conflict_type = "no_clear_conflict"


        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        hybrid_score = (
            self.calculate_hybrid_score(

                contradiction_probability,

                structured_score,

                temporal_score
            )
        )


        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        is_conflict = (

            hybrid_score
            >=
            self.conflict_threshold
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        level = (
            self.confidence_level(
                hybrid_score
            )
        )


        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        explanation = (
            self.generate_explanation(

                nli_label,

                contradiction_probability,

                structured_score,

                structured_type,

                temporal_score,

                temporal_type
            )
        )


        return {

            "nli_label":
                nli_label,

            "nli_contradiction_probability":
                contradiction_probability,

            "structured_conflict_score":
                structured_score,

            "temporal_conflict_score":
                temporal_score,

            "temporal_conflict_type":
                temporal_type,

            "hybrid_conflict_score":
                hybrid_score,

            "is_conflict":
                is_conflict,

            "confidence_level":
                level,

            "conflict_type":
                final_conflict_type,

            "explanation":
                explanation
        }