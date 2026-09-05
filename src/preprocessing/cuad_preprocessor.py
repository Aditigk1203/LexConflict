from typing import List, Dict, Any

from .clause_preprocessor import preprocess_document


def extract_cuad_documents(cuad_data: Dict[str, Any]):
    """
    Extract document-like objects from the SQuAD-style CUAD format.
    """

    documents = []

    articles = cuad_data.get(
        "data",
        []
    )

    for article in articles:

        title = article.get(
            "title",
            "unknown"
        )

        paragraphs = article.get(
            "paragraphs",
            []
        )

        for paragraph_index, paragraph in enumerate(
            paragraphs
        ):

            context = paragraph.get(
                "context",
                ""
            )

            if not context:
                continue

            document = {
                "id": (
                    f"{title}_paragraph_"
                    f"{paragraph_index + 1}"
                ),
                "text": context,
            }

            documents.append(document)

    return documents


def preprocess_cuad(
    cuad_data: Dict[str, Any]
) -> List:

    documents = extract_cuad_documents(
        cuad_data
    )

    all_clauses = []

    for document in documents:

        clauses = preprocess_document(
            document,
            dataset="cuad"
        )

        all_clauses.extend(
            clauses
        )

    return all_clauses