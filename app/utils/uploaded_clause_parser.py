import re
from typing import List

from src.preprocessing.clause_schema import Clause
from src.preprocessing.clause_preprocessor import (
    clean_text,
    detect_party,
    detect_modality,
    detect_condition,
    extract_proposition,
)


# ============================================================
# HEADING DETECTION
# ============================================================

def _is_heading(line: str) -> bool:
    """
    Detect obvious section headings such as:

        PAYMENT TERMS
        PAYMENT EXTENSION
        CONFIDENTIALITY

    These are not treated as independent clauses.
    """

    text = line.strip()

    if not text:
        return False

    if len(text) > 100:
        return False

    # A heading normally does not end with sentence punctuation.
    if re.search(r"[.!?;:]$", text):
        return False

    # Do not classify numbered clauses as headings.
    if re.match(
        r"^(?:\(?\d+(?:\.\d+)*\)?[.)]?|[A-Za-z][.)])\s+",
        text
    ):
        return False

    letters = re.sub(
        r"[^A-Za-z]",
        "",
        text
    )

    if not letters:
        return False

    uppercase_ratio = (
        sum(c.isupper() for c in letters)
        / len(letters)
    )

    # Short mostly-uppercase text is treated as a heading.
    return (
        uppercase_ratio >= 0.80
        and len(text.split()) <= 12
    )


# ============================================================
# EXPLICIT CLAUSE NUMBER DETECTION
# ============================================================

def _starts_explicit_clause(line: str) -> bool:
    """
    Detect:

        1. Payment...
        1.1 Payment...
        (1) Payment...
        a) Payment...
    """

    return bool(
        re.match(
            r"^(?:"
            r"\d+(?:\.\d+)*[.)]"
            r"|\([0-9]+\)"
            r"|[a-zA-Z][.)]"
            r")\s+",
            line.strip()
        )
    )


# ============================================================
# SENTENCE END DETECTION
# ============================================================

def _has_sentence_end(text: str) -> bool:

    return bool(
        re.search(
            r"[.!?][\"')\]]*$",
            text.strip()
        )
    )


# ============================================================
# LINE NORMALIZATION
# ============================================================

def _normalize_line(line: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        line
    ).strip()


# ============================================================
# MAIN PARSER
# ============================================================

def parse_uploaded_contract(
    text: str,
    document_id: str,
    dataset: str = "uploaded_contract"
) -> List[Clause]:
    """
    Parse uploaded contract text into complete clauses.

    Important differences from the original ContractNLI
    preprocessing:

    1. Wrapped lines are joined.
    2. Obvious section headings are ignored.
    3. Complete sentences are preserved.
    4. Legal metadata is extracted for each clause.
    """

    if not text or not text.strip():
        return []

    # --------------------------------------------------------
    # Normalize newlines
    # --------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    raw_lines = [

        _normalize_line(line)

        for line in text.split("\n")
    ]


    # --------------------------------------------------------
    # Build complete clause blocks
    # --------------------------------------------------------

    blocks = []

    current = []


    def flush_current():

        nonlocal current

        if not current:
            return

        value = " ".join(
            current
        ).strip()

        value = clean_text(
            value
        )

        # Ignore tiny fragments.
        if len(value) >= 15:

            blocks.append(
                value
            )

        current = []


    for line in raw_lines:

        # Empty line normally separates sections.
        if not line:

            flush_current()

            continue


        # Ignore section headings.
        if _is_heading(line):

            flush_current()

            continue


        # If a new explicit clause starts,
        # close the previous clause first.
        if (
            _starts_explicit_clause(line)
            and current
        ):

            flush_current()


        current.append(
            line
        )


        combined = " ".join(
            current
        )


        # Close once the complete sentence ends.
        if _has_sentence_end(
            combined
        ):

            flush_current()


    flush_current()


    # --------------------------------------------------------
    # Convert blocks into Clause objects
    # --------------------------------------------------------

    clauses = []


    for index, clause_text in enumerate(
        blocks,
        start=1
    ):

        clause = Clause(

            document_id=document_id,

            clause_id=(
                f"{document_id}_clause_"
                f"{index:04d}"
            ),

            text=clause_text,

            party=detect_party(
                clause_text
            ),

            modality=detect_modality(
                clause_text
            ),

            condition=detect_condition(
                clause_text
            ),

            proposition=extract_proposition(
                clause_text
            ),

            dataset=dataset,

            context_text=clause_text,
        )


        clauses.append(
            clause
        )


    return clauses