from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class RiskResult:
    """
    Stores risk information for a single clause/node.
    """

    node_id: str

    direct_conflict_score: float
    propagated_conflict_score: float

    conflict_degree: int
    conflict_density: float

    severity_score: float

    risk_score: float
    risk_level: str

    risk_factors: Dict[str, Any]

    explanation: str

    def to_dict(self):
        return asdict(self)