from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphEdge:

    source_id: str
    target_id: str

    relationship: str

    confidence: float

    conflict_type: Optional[str] = None

    nli_label: Optional[str] = None
    nli_contradiction_probability: float = 0.0

    structured_conflict_score: float = 0.0

    hybrid_conflict_score: float = 0.0

    explanation: Optional[str] = None