from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class Clause:
    """
    Standard representation of a legal contract clause.
    """

    document_id: str
    clause_id: str
    text: str

    party: Optional[str] = None
    modality: Optional[str] = None
    condition: Optional[str] = None
    proposition: Optional[str] = None

    dataset: Optional[str] = None
    source_label: Optional[str] = None

    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # Parent/context information for legal sub-clauses.
    parent_clause_id: Optional[str] = None
    context_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Clause object into a dictionary.
        """
        return asdict(self)