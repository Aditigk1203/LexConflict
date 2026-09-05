from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class ConflictResult:
    """
    Result produced by the LexConflict conflict engine.
    """

    clause_a_id: str
    clause_b_id: str

    clause_a_text: str
    clause_b_text: str

    is_conflict: bool

    confidence: float

    conflict_type: str

    semantic_similarity: float

    modality_a: Optional[str]
    modality_b: Optional[str]

    modality_conflict: float

    negation_a: bool
    negation_b: bool

    negation_conflict: float

    same_party: float

    condition_similarity: float

    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result into a dictionary.
        """

        return asdict(self)