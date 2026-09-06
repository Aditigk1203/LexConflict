from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class ExplanationResult:
    """
    Structured explanation for a LexConflict clause.
    """

    node_id: str
    clause_text: str

    risk_score: float
    risk_level: str

    direct_conflict_score: float
    propagated_conflict_score: float

    conflict_degree: int
    conflict_density: float
    severity_score: float

    conflict_types: List[str]
    conflict_neighbors: List[str]

    hotspot_level: str

    contributing_factors: Dict
    explanation: str
    recommendation: str

    def to_dict(self) -> Dict:
        """
        Convert explanation result into a JSON-serializable dictionary.
        """
        return asdict(self)