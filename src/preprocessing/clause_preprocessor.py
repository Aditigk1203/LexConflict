import re
from typing import List, Dict, Any, Optional

from .clause_schema import Clause


# ---------------------------------------------------------
# 1. TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic cleaning of contract text.
    """

    if not text:
        return ""

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize repeated newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


# ---------------------------------------------------------
# 2. SENTENCE / CLAUSE SEGMENTATION
# ---------------------------------------------------------

def split_into_clauses(text: str) -> List[Dict[str, Any]]:
    """
    Split a contract into reasonably sized clause units.

    This is a lightweight first version.
    We will improve this later if needed.
    """

    if not text:
        return []

    clauses = []

    # Split at:
    # - sentence-ending punctuation
    # - newlines
    #
    # Keep the punctuation with the clause.

    pattern = r".+?(?:[.!?](?=\s|$)|\n|$)"

    matches = re.finditer(pattern, text, flags=re.DOTALL)

    for match in matches:

        clause_text = match.group(0).strip()

        if not clause_text:
            continue

        # Ignore extremely short fragments
        if len(clause_text) < 10:
            continue

        clauses.append(
            {
                "text": clause_text,
                "start_char": match.start(),
                "end_char": match.end(),
            }
        )

    return clauses


# ---------------------------------------------------------
# 3. MODALITY DETECTION
# ---------------------------------------------------------

def detect_modality(text: str) -> Optional[str]:
    """
    Detect common legal modality patterns.
    """

    text_lower = text.lower()

    # Prohibition
    prohibition_patterns = [
        r"\bshall not\b",
        r"\bmust not\b",
        r"\bmay not\b",
        r"\bprohibited\b",
        r"\bprohibit(ed|s)?\b",
        r"\bforbidden\b",
        r"\bnot permitted\b",
    ]

    for pattern in prohibition_patterns:
        if re.search(pattern, text_lower):
            return "prohibition"

    # Obligation
    obligation_patterns = [
        r"\bshall\b",
        r"\bmust\b",
        r"\brequired to\b",
        r"\bobligated to\b",
        r"\bwill\b",
    ]

    for pattern in obligation_patterns:
        if re.search(pattern, text_lower):
            return "obligation"

    # Permission
    permission_patterns = [
        r"\bmay\b",
        r"\bpermitted to\b",
        r"\bhas the right to\b",
        r"\bis entitled to\b",
        r"\bcan\b",
    ]

    for pattern in permission_patterns:
        if re.search(pattern, text_lower):
            return "permission"

    # Recommendation / weaker modality
    recommendation_patterns = [
        r"\bshould\b",
        r"\brecommended\b",
        r"\bencouraged to\b",
    ]

    for pattern in recommendation_patterns:
        if re.search(pattern, text_lower):
            return "recommendation"

    return None


# ---------------------------------------------------------
# 4. PARTY DETECTION
# ---------------------------------------------------------

def detect_party(text: str) -> Optional[str]:
    """
    Detect common party references.
    """

    text_lower = text.lower()

    party_patterns = {
        "supplier": [
            r"\bsupplier\b",
            r"\bvendor\b",
            r"\bservice provider\b",
        ],

        "customer": [
            r"\bcustomer\b",
            r"\bclient\b",
            r"\bbuyer\b",
        ],

        "company": [
            r"\bcompany\b",
            r"\bcorporation\b",
        ],

        "employee": [
            r"\bemployee\b",
            r"\bpersonnel\b",
        ],

        "contractor": [
            r"\bcontractor\b",
        ],

        "licensor": [
            r"\blicensor\b",
        ],

        "licensee": [
            r"\blicensee\b",
        ],

        "party": [
            r"\bparty\b",
            r"\bparties\b",
        ],
    }

    for party, patterns in party_patterns.items():

        for pattern in patterns:

            if re.search(pattern, text_lower):
                return party

    return None


# ---------------------------------------------------------
# 5. CONDITION DETECTION
# ---------------------------------------------------------

def detect_condition(text: str) -> Optional[str]:
    """
    Detect conditional language in legal clauses.
    """

    condition_patterns = [
        r"\bif\b[^.;]+",
        r"\bunless\b[^.;]+",
        r"\bprovided that\b[^.;]+",
        r"\bsubject to\b[^.;]+",
        r"\bin the event that\b[^.;]+",
        r"\bwhen\b[^.;]+",
        r"\bwhere\b[^.;]+",
    ]

    for pattern in condition_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            return match.group(0).strip()

    return None


# ---------------------------------------------------------
# 6. PROPOSITION EXTRACTION
# ---------------------------------------------------------

def extract_proposition(text: str) -> str:
    """
    Lightweight proposition extraction.

    At this stage we keep the clause text but remove
    some common modality words.

    Later this can be replaced with a more sophisticated
    legal semantic representation.
    """

    proposition = text

    patterns = [
        r"\bshall not\b",
        r"\bmust not\b",
        r"\bmay not\b",
        r"\bshall\b",
        r"\bmust\b",
        r"\bmay\b",
        r"\bshould\b",
        r"\bwill\b",
        r"\bcan\b",
    ]

    for pattern in patterns:

        proposition = re.sub(
            pattern,
            "",
            proposition,
            flags=re.IGNORECASE
        )

    proposition = re.sub(
        r"\s+",
        " ",
        proposition
    ).strip()

    return proposition


# ---------------------------------------------------------
# 7. PROCESS A DOCUMENT
# ---------------------------------------------------------

def preprocess_document(
    document: Dict[str, Any],
    dataset: str = "unknown"
) -> List[Clause]:
    """
    Convert a document into standardized Clause objects.

    Parameters
    ----------
    document:
        Dictionary containing document information.

    dataset:
        Dataset name, e.g. "contractnli" or "cuad".

    Returns
    -------
    List[Clause]
    """

    # Try common document ID fields
    document_id = (
        document.get("id")
        or document.get("document_id")
        or document.get("title")
        or "unknown_document"
    )

    # Try common text fields
    text = (
        document.get("text")
        or document.get("context")
        or ""
    )

    text = clean_text(text)

    if not text:
        return []

    raw_clauses = split_into_clauses(text)

    processed_clauses = []

    for index, item in enumerate(raw_clauses):

        clause_text = item["text"]

        clause = Clause(
            document_id=str(document_id),

            clause_id=(
                f"{document_id}_clause_{index + 1:04d}"
            ),

            text=clause_text,

            party=detect_party(clause_text),

            modality=detect_modality(clause_text),

            condition=detect_condition(clause_text),

            proposition=extract_proposition(clause_text),

            dataset=dataset,

            start_char=item["start_char"],

            end_char=item["end_char"],
        )

        processed_clauses.append(clause)

    return processed_clauses