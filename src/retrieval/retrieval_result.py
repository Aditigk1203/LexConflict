from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RetrievalResult:
    """
    Represents one retrieved candidate clause.
    """

    query_clause_id: str
    candidate_clause_id: str

    score: float

    query_text: str
    candidate_text: str

    query_document_id: str
    candidate_document_id: str

    rank: int

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.
        """

        return {
            "query_clause_id": self.query_clause_id,
            "candidate_clause_id": self.candidate_clause_id,
            "score": self.score,
            "query_text": self.query_text,
            "candidate_text": self.candidate_text,
            "query_document_id": self.query_document_id,
            "candidate_document_id": self.candidate_document_id,
            "rank": self.rank,
        }