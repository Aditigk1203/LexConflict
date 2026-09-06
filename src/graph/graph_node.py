from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphNode:

    node_id: str
    document_id: str
    clause_id: str
    text: str

    party: Optional[str] = None
    modality: Optional[str] = None
    condition: Optional[str] = None
    proposition: Optional[str] = None

    dataset: Optional[str] = None

    # Number of conflict relationships
    conflict_degree: int = 0