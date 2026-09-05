import re
from typing import Any, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ConflictFeatureExtractor:

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Generic helpers
    # ---------------------------------------------------------

    @staticmethod
    def get_value(clause: Any, key: str, default=None):
        if isinstance(clause, dict):
            return clause.get(key, default)

        return getattr(clause, key, default)

    def get_text(self, clause: Any) -> str:
        return str(self.get_value(clause, "text", "") or "")

    def get_clause_id(self, clause: Any) -> str:
        return str(self.get_value(clause, "clause_id", "") or "")

    # ---------------------------------------------------------
    # Semantic similarity
    # ---------------------------------------------------------

    def semantic_similarity(
        self,
        clause_a: Any,
        clause_b: Any
    ) -> float:

        text_a = self.get_text(clause_a)
        text_b = self.get_text(clause_b)

        if not text_a or not text_b:
            return 0.0

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform([
            text_a,
            text_b
        ])

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return float(score)

    # ---------------------------------------------------------
    # Negation detection
    # ---------------------------------------------------------

    def contains_negation(self, text: str) -> bool:

        text = text.lower()

        patterns = [
            r"\bnot\b",
            r"\bno\b",
            r"\bnever\b",
            r"\bneither\b",
            r"\bnor\b",
            r"\bwithout\b",
            r"\bprohibited\b",
            r"\bprohibit\b",
            r"\bforbidden\b",
            r"\bshall not\b",
            r"\bmay not\b"
        ]

        return any(
            re.search(pattern, text)
            for pattern in patterns
        )

    # ---------------------------------------------------------
    # Modality conflict
    # ---------------------------------------------------------

    def modality_conflict(
        self,
        modality_a: Optional[str],
        modality_b: Optional[str]
    ) -> float:

        if not modality_a or not modality_b:
            return 0.0

        a = modality_a.lower()
        b = modality_b.lower()

        conflict_pairs = {
            ("obligation", "prohibition"),
            ("prohibition", "obligation"),
            ("permission", "prohibition"),
            ("prohibition", "permission")
        }

        if (a, b) in conflict_pairs:
            return 1.0

        return 0.0

    # ---------------------------------------------------------
    # Negation conflict
    # ---------------------------------------------------------

    def negation_conflict(
        self,
        text_a: str,
        text_b: str
    ) -> float:

        neg_a = self.contains_negation(text_a)
        neg_b = self.contains_negation(text_b)

        if neg_a != neg_b:
            return 1.0

        return 0.0

    # ---------------------------------------------------------
    # Party compatibility
    # ---------------------------------------------------------

    def same_party(
        self,
        clause_a: Any,
        clause_b: Any
    ) -> float:

        party_a = self.get_value(clause_a, "party")
        party_b = self.get_value(clause_b, "party")

        if not party_a or not party_b:
            return 0.0

        return float(
            str(party_a).lower() == str(party_b).lower()
        )

    # ---------------------------------------------------------
    # Condition similarity
    # ---------------------------------------------------------

    def condition_similarity(
        self,
        clause_a: Any,
        clause_b: Any
    ) -> float:

        condition_a = self.get_value(
            clause_a,
            "condition",
            ""
        ) or ""

        condition_b = self.get_value(
            clause_b,
            "condition",
            ""
        ) or ""

        if not condition_a or not condition_b:
            return 0.0

        vectorizer = TfidfVectorizer()

        matrix = vectorizer.fit_transform([
            condition_a,
            condition_b
        ])

        return float(
            cosine_similarity(
                matrix[0:1],
                matrix[1:2]
            )[0][0]
        )

    # ---------------------------------------------------------
    # Full feature extraction
    # ---------------------------------------------------------

    def extract_features(
        self,
        clause_a: Any,
        clause_b: Any,
        semantic_similarity: Optional[float] = None
    ) -> Dict[str, float]:

        text_a = self.get_text(clause_a)
        text_b = self.get_text(clause_b)

        modality_a = self.get_value(
            clause_a,
            "modality"
        )

        modality_b = self.get_value(
            clause_b,
            "modality"
        )

        if semantic_similarity is None:
            semantic_similarity = self.semantic_similarity(
                clause_a,
                clause_b
            )

        return {
            "semantic_similarity": float(
                semantic_similarity
            ),

            "modality_conflict": self.modality_conflict(
                modality_a,
                modality_b
            ),

            "negation_conflict": self.negation_conflict(
                text_a,
                text_b
            ),

            "same_party": self.same_party(
                clause_a,
                clause_b
            ),

            "condition_similarity": self.condition_similarity(
                clause_a,
                clause_b
            )
        }