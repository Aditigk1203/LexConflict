import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "kaggle"
    / "contract-nli"
    / "dev.json"
)

CLAUSES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dev_clauses.json"
)

INDEXED_CLAUSE_FIELD = "context_text"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval"
    / "dev_tfidf_context_evidence_retrieval.json"
)
K_VALUES = [1, 2, 5, 10]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def interval_overlaps(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def build_gold_evidence_by_document(
    raw_documents: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Set[int]]]:
    """
    Output:
        {
          document_id: {
            hypothesis_id: {gold_span_index, ...}
          }
        }

    ContractNLI annotations reference entries in document["spans"]
    by index. Empty span lists are NotMentioned examples and are
    excluded from evidence-retrieval scoring.
    """
    gold_by_document = {}

    for document in raw_documents:
        document_id = str(document["id"])
        annotations = {}

        for annotation_set in document.get("annotation_sets", []):
            annotations.update(
                annotation_set.get("annotations", {})
            )

        gold_by_hypothesis = {}

        for hypothesis_id, annotation in annotations.items():
            gold_span_indices = set(
                annotation.get("spans", [])
            )

            if gold_span_indices:
                gold_by_hypothesis[
                    hypothesis_id
                ] = gold_span_indices

        gold_by_document[
            document_id
        ] = gold_by_hypothesis

    return gold_by_document


def build_hypothesis_lookup(
    raw_data: Dict[str, Any],
) -> Dict[str, str]:
    """
    ContractNLI labels are stored as:
    {
        "nda-1": {
            "short_description": "...",
            "hypothesis": "..."
        }
    }
    """
    labels = raw_data["labels"]

    if isinstance(labels, dict):
        return {
            str(hypothesis_id): str(
                label_data.get("hypothesis", "")
                if isinstance(label_data, dict)
                else label_data
            )
            for hypothesis_id, label_data in labels.items()
        }

    return {
        str(item["id"]): str(item["hypothesis"])
        for item in labels
    }
def gold_character_ranges(
    document: Dict[str, Any],
    gold_span_indices: Set[int],
) -> List[List[int]]:
    spans = document["spans"]

    return [
        spans[index]
        for index in gold_span_indices
        if 0 <= index < len(spans)
    ]


def relevant_clause_ids(
    clauses: List[Dict[str, Any]],
    gold_ranges: List[List[int]],
) -> Set[str]:
    """
    A clause is relevant when its source character range overlaps at
    least one annotated ContractNLI evidence span.
    """
    relevant = set()

    for clause in clauses:
        clause_start = clause.get("start_char")
        clause_end = clause.get("end_char")

        if clause_start is None or clause_end is None:
            continue

        for span_start, span_end in gold_ranges:
            if interval_overlaps(
                int(clause_start),
                int(clause_end),
                int(span_start),
                int(span_end),
            ):
                relevant.add(str(clause["clause_id"]))
                break

    return relevant


def rank_clauses(
    hypothesis: str,
    clauses: List[Dict[str, Any]],
    indexed_clause_field: str,
) -> List[Dict[str, Any]]:

    texts = [
        str(
            clause.get(indexed_clause_field)
            or clause.get("text", "")
        )
        for clause in clauses
    ]

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )

    clause_matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([hypothesis])

    scores = cosine_similarity(
        query_vector,
        clause_matrix,
    )[0]

    ranked_indices = np.argsort(-scores)

    return [
        {
            "clause_id": str(clauses[index]["clause_id"]),
            "score": float(scores[index]),
            "text": clauses[index].get("text", ""),
        }
        for index in ranked_indices
    ]

def main():
    raw_data = load_json(RAW_DATA_PATH)
    processed_clauses = load_json(CLAUSES_PATH)

    documents = raw_data["documents"]

    raw_document_by_id = {
        str(document["id"]): document
        for document in documents
    }

    clauses_by_document = defaultdict(list)

    for clause in processed_clauses:
        clauses_by_document[
            str(clause["document_id"])
        ].append(clause)

    hypothesis_lookup = build_hypothesis_lookup(raw_data)

    gold_by_document = build_gold_evidence_by_document(
        documents
    )

    recall_hits = {
        k: 0
        for k in K_VALUES
    }

    reciprocal_ranks = []
    evaluated_queries = 0
    skipped_missing_clause_alignment = 0
    query_results = []

    for document_id, gold_by_hypothesis in (
        gold_by_document.items()
    ):
        document = raw_document_by_id[document_id]
        document_clauses = clauses_by_document[document_id]

        if not document_clauses:
            continue

        for hypothesis_id, gold_span_indices in (
            gold_by_hypothesis.items()
        ):
            hypothesis = hypothesis_lookup.get(hypothesis_id)

            if not hypothesis:
                continue

            gold_ranges = gold_character_ranges(
                document,
                gold_span_indices,
            )

            gold_clause_ids = relevant_clause_ids(
                document_clauses,
                gold_ranges,
            )

            if not gold_clause_ids:
                skipped_missing_clause_alignment += 1
                continue

            ranked_results = rank_clauses(
                hypothesis=hypothesis,
                clauses=document_clauses,
                indexed_clause_field=INDEXED_CLAUSE_FIELD,
            )

            ranked_clause_ids = [
                result["clause_id"]
                for result in ranked_results
            ]

            evaluated_queries += 1

            first_relevant_rank = None

            for rank, clause_id in enumerate(
                ranked_clause_ids,
                start=1,
            ):
                if clause_id in gold_clause_ids:
                    first_relevant_rank = rank
                    break

            if first_relevant_rank is not None:
                reciprocal_ranks.append(
                    1 / first_relevant_rank
                )
            else:
                reciprocal_ranks.append(0.0)

            for k in K_VALUES:
                if any(
                    clause_id in gold_clause_ids
                    for clause_id in ranked_clause_ids[:k]
                ):
                    recall_hits[k] += 1

            query_results.append(
                {
                    "document_id": document_id,
                    "hypothesis_id": hypothesis_id,
                    "hypothesis": hypothesis,
                    "gold_span_indices": sorted(
                        gold_span_indices
                    ),
                    "gold_clause_ids": sorted(
                        gold_clause_ids
                    ),
                    "first_relevant_rank": first_relevant_rank,
                    "top_10": ranked_results[:10],
                }
            )

    if evaluated_queries == 0:
        raise RuntimeError(
            "No evaluable queries found. Check clause "
            "start_char/end_char alignment."
        )

    metrics = {
        f"recall_at_{k}": round(
            recall_hits[k] / evaluated_queries,
            4,
        )
        for k in K_VALUES
    }

    metrics["mrr"] = round(
        sum(reciprocal_ranks) / evaluated_queries,
        4,
    )

    output = {
        "configuration": {
            "dataset_split": "dev",
            "retrieval_method": "tfidf",
            "tfidf": {
                "max_features": 20000,
                "ngram_range": [1, 2],
                "min_df": 1,
                "sublinear_tf": True,
            },
            "indexed_clause_field": INDEXED_CLAUSE_FIELD,
            "k_values": K_VALUES,
        },
        "counts": {
            "evaluated_queries": evaluated_queries,
            "skipped_missing_clause_alignment":
                skipped_missing_clause_alignment,
        },
        "metrics": metrics,
        "query_results": query_results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTF-IDF EVIDENCE RETRIEVAL BASELINE")
    print(f"Evaluated queries: {evaluated_queries}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nSaved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

