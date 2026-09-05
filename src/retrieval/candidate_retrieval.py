from typing import Any, Dict, List, Sequence, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.representation.tfidf_representation import (
    TfidfClauseRepresentation,
)

from src.retrieval.retrieval_result import (
    RetrievalResult,
)


ClauseLike = Union[Dict[str, Any], Any]


class CandidateRetriever:
    """
    TF-IDF based candidate clause retriever.

    The retriever:
        1. Builds TF-IDF representations for clauses.
        2. Computes cosine similarity.
        3. Removes the query clause itself.
        4. Optionally restricts retrieval to the same document.
        5. Returns top-k candidate clauses.
    """

    def __init__(
        self,
        top_k: int = 10,
        same_document_only: bool = True,
    ):

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        self.top_k = top_k

        self.same_document_only = (
            same_document_only
        )

        self.representation = (
            TfidfClauseRepresentation()
        )

        self.clauses = []

        self.clause_ids = []

        self.fitted = False

    # --------------------------------------------------
    # Helper methods
    # --------------------------------------------------

    @staticmethod
    def _get_value(
        clause: ClauseLike,
        key: str,
        default=None,
    ):
        """
        Support both:
            - Clause dataclass objects
            - dictionaries
        """

        if isinstance(clause, dict):
            return clause.get(
                key,
                default
            )

        return getattr(
            clause,
            key,
            default
        )

    @classmethod
    def _get_text(
        cls,
        clause: ClauseLike,
    ) -> str:

        text = cls._get_value(
            clause,
            "text",
            "",
        )

        return str(text or "")

    @classmethod
    def _get_id(
        cls,
        clause: ClauseLike,
    ) -> str:

        clause_id = cls._get_value(
            clause,
            "clause_id",
            None,
        )

        if clause_id is None:
            raise ValueError(
                "Every clause must have a clause_id."
            )

        return str(clause_id)

    @classmethod
    def _get_document_id(
        cls,
        clause: ClauseLike,
    ) -> str:

        document_id = cls._get_value(
            clause,
            "document_id",
            "",
        )

        return str(
            document_id or ""
        )

    # --------------------------------------------------
    # Fit retriever
    # --------------------------------------------------

    def fit(
        self,
        clauses: Sequence[ClauseLike],
    ):
        """
        Build the retrieval index.
        """

        if not clauses:
            raise ValueError(
                "Cannot fit retriever on empty clauses."
            )

        self.clauses = list(clauses)

        texts = [
            self._get_text(clause)
            for clause in self.clauses
        ]

        self.representation.fit_transform(
            texts
        )

        self.clause_ids = [
            self._get_id(clause)
            for clause in self.clauses
        ]

        self.fitted = True

        return self

    # --------------------------------------------------
    # Retrieve candidates
    # --------------------------------------------------

    def retrieve(
        self,
        query_clause: ClauseLike,
        top_k: int = None,
        same_document_only: bool = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most similar clauses for one query clause.
        """

        if not self.fitted:
            raise RuntimeError(
                "Retriever has not been fitted."
            )

        if top_k is None:
            top_k = self.top_k

        if same_document_only is None:
            same_document_only = (
                self.same_document_only
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_id = self._get_id(
            query_clause
        )

        query_document_id = (
            self._get_document_id(
                query_clause
            )
        )

        query_text = self._get_text(
            query_clause
        )

        # Convert query into TF-IDF vector
        query_vector = (
            self.representation.transform(
                [query_text]
            )
        )

        # Calculate cosine similarity against
        # every indexed clause.
        similarities = cosine_similarity(
            query_vector,
            self.representation.matrix,
        )[0]

        candidates = []

        for index, score in enumerate(
            similarities
        ):

            candidate = self.clauses[index]

            candidate_id = (
                self._get_id(candidate)
            )

            candidate_document_id = (
                self._get_document_id(
                    candidate
                )
            )

            # Don't retrieve the clause itself.
            if candidate_id == query_id:
                continue

            # For conflict detection we normally
            # want clauses from the same contract.
            if (
                same_document_only
                and candidate_document_id
                != query_document_id
            ):
                continue

            candidates.append(
                (
                    float(score),
                    index,
                )
            )

        # Highest similarity first
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # Keep only top-k
        candidates = candidates[:top_k]

        results = []

        for rank, (score, index) in enumerate(
            candidates,
            start=1,
        ):

            candidate = self.clauses[index]

            results.append(
                RetrievalResult(
                    query_clause_id=query_id,

                    candidate_clause_id=(
                        self._get_id(candidate)
                    ),

                    score=score,

                    query_text=query_text,

                    candidate_text=(
                        self._get_text(candidate)
                    ),

                    query_document_id=(
                        query_document_id
                    ),

                    candidate_document_id=(
                        self._get_document_id(
                            candidate
                        )
                    ),

                    rank=rank,
                )
            )

        return results

    # --------------------------------------------------
    # Retrieve for every clause
    # --------------------------------------------------

    def retrieve_all(
        self,
        top_k: int = None,
        same_document_only: bool = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve candidates for every indexed clause.
        """

        if not self.fitted:
            raise RuntimeError(
                "Retriever has not been fitted."
            )

        results = []

        for clause in self.clauses:

            clause_results = self.retrieve(
                clause,
                top_k=top_k,
                same_document_only=(
                    same_document_only
                ),
            )

            results.extend(
                clause_results
            )

        return results
    
    def retrieve_candidates(
        clause,
        all_clauses,
        top_k=10,
        same_document_only=True,
    ):
        """
        Convenience function for one-shot retrieval.
    
        Example
        -------
        candidates = retrieve_candidates(
            clause,
            all_clauses,
            top_k=10
        )
        """

        retriever = CandidateRetriever(
            top_k=top_k,
            same_document_only=(
                same_document_only
            ),
        )
    
        retriever.fit(
            all_clauses
        )

        return retriever.retrieve(
            clause,
            top_k=top_k,
            same_document_only=(
                same_document_only
            ),
        )