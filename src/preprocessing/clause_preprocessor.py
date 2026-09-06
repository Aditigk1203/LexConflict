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

def is_subclause_fragment(text: str) -> bool:
    """
    Detect common legal list/sub-clause fragments.

    Examples:
        a) ...
        b) ...
        (c) ...
        1. ...
        (1) ...
    """

    text = text.strip()

    patterns = [
        r"^[a-z]\)",
        r"^\([a-z]\)",
        r"^[ivxlcdm]+\)",
        r"^\([ivxlcdm]+\)",
        r"^\d+\.",
        r"^\(\d+\)",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )
        
def add_clause_context(
    clauses: List[Dict[str, Any]],
    document_id: str
) -> List[Dict[str, Any]]:
    """
    Attach parent-clause context to legal sub-clause fragments.

    The original clause text is preserved.

    Example:

        Parent:
        The obligation of confidentiality shall not apply to any
        Confidential Information that:

        Child:
        a) was known to the Receiving Party prior to such disclosure...

    The child receives the parent clause ID and combined context text.
    """

    enriched = []

    current_parent = None

    for index, clause in enumerate(clauses):

        clause = dict(clause)

        clause_id = f"{document_id}_clause_{index + 1:04d}"
        clause["clause_id"] = clause_id

        text = clause["text"]

        if is_subclause_fragment(text) and current_parent is not None:

            clause["parent_clause_id"] = current_parent["clause_id"]

            clause["context_text"] = (
                current_parent["text"].rstrip()
                + " "
                + text.lstrip()
            )

        else:

            clause["parent_clause_id"] = None
            clause["context_text"] = text

            # A normal clause becomes the potential parent
            # for subsequent list fragments.
            current_parent = clause

        enriched.append(clause)

    return enriched

# ---------------------------------------------------------
# 3. MODALITY DETECTION
# ---------------------------------------------------------

def detect_modality(text: str) -> Optional[str]:
    """
    Detect common legal modality patterns.

    The detector checks:
    1. Definition-style language first
    2. Prohibition
    3. Permission
    4. Obligation

    Definition clauses are separated from obligations because
    expressions such as "shall mean" and "will have the meaning"
    are definitional rather than normative obligations.
    """

    text_lower = text.lower().strip()

    # --------------------------------------------------------------
    # Definition / terminology
    # --------------------------------------------------------------
    definition_patterns = [
        r"\bshall mean\b",
        r"\bshall have the meaning\b",
        r"\bwill mean\b",
        r"\bwill have the meaning\b",
        r"\bwill have the respective meanings\b",
        r"\bmeans\b",
        r"\bis defined as\b",
        r"\bare defined as\b",
        r"\brefers to\b",
        r"\bshall be defined as\b",
    ]

    for pattern in definition_patterns:
        if re.search(pattern, text_lower):
            return "definition"

    # --------------------------------------------------------------
    # Prohibition
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Permission
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Obligation
    # --------------------------------------------------------------
    obligation_patterns = [
        r"\bshall\b",
        r"\bmust\b",
        r"\brequired to\b",
        r"\bobligated to\b",
    ]

    for pattern in obligation_patterns:
        if re.search(pattern, text_lower):
            return "obligation"

    # --------------------------------------------------------------
    # No clear modality
    # --------------------------------------------------------------
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

    # Add parent/context information.
    raw_clauses = add_clause_context(
        raw_clauses,
        str(document_id)
    )

    processed_clauses = []

    for item in raw_clauses:

        clause_text = item["text"]

        clause = Clause(
            document_id=str(document_id),

            clause_id=item["clause_id"],

            text=clause_text,

            party=detect_party(clause_text),

            modality=detect_modality(clause_text),

            condition=detect_condition(clause_text),

            proposition=extract_proposition(clause_text),

            dataset=dataset,

            start_char=item["start_char"],

            end_char=item["end_char"],

            parent_clause_id=item.get("parent_clause_id"),

            context_text=item.get(
                "context_text",
                clause_text
            ),
        )

        processed_clauses.append(clause)

    return processed_clauses