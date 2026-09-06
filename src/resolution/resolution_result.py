from dataclasses import dataclass


@dataclass
class ResolutionResult:

    severity: str

    recommendation: str

    explanation: str

    suggested_revision: str