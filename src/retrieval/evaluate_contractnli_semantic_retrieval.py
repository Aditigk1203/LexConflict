import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lexconflict_legalbert"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval"
    / "dev_legalbert_semantic_evidence_retrieval.json"
)
SCORES_OUTPUT_PATH = (
    Path("results")
    / "retrieval"
    / "dev_legalbert_semantic_scores.json"
)
INDEXED_CLAUSE_FIELD = "text"

K_VALUES = [1, 2, 5, 10]
BATCH_SIZE = 16
MAX_LENGTH = 256


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


def build_hypothesis_lookup(
    raw_data: Dict[str, Any],
) -> Dict[str, str]:
    labels = raw_data["labels"]

    return {
        str(hypothesis_id): str(
            label_data.get("hypothesis", "")
            if isinstance(label_data, dict)
            else label_data
        )
        for hypothesis_id, label_data in labels.items()
    }


def build_gold_evidence_by_document(
    raw_documents: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Set[int]]]:
    gold_by_document = {}

    for document in raw_documents:
        annotations = {}

        for annotation_set in document.get(
            "annotation_sets",
            [],
        ):
            annotations.update(
                annotation_set.get("annotations", {})
            )

        gold_by_hypothesis = {
            hypothesis_id: set(
                annotation.get("spans", [])
            )
            for hypothesis_id, annotation in annotations.items()
            if annotation.get("spans", [])
        }

        gold_by_document[
            str(document["id"])
        ] = gold_by_hypothesis

    return gold_by_document


def relevant_clause_ids(
    clauses: List[Dict[str, Any]],
    gold_ranges: List[List[int]],
) -> Set[str]:
    relevant = set()

    for clause in clauses:
        clause_start = clause.get("start_char")
        clause_end = clause.get("end_char")

        if clause_start is None or clause_end is None:
            continue

        for evidence_start, evidence_end in gold_ranges:
            if interval_overlaps(
                int(clause_start),
                int(clause_end),
                int(evidence_start),
                int(evidence_end),
            ):
                relevant.add(str(clause["clause_id"]))
                break

    return relevant


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    summed_embeddings = torch.sum(
        token_embeddings * mask,
        dim=1,
    )

    token_counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9,
    )

    return summed_embeddings / token_counts


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device,
) -> torch.Tensor:
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[
                start:start + BATCH_SIZE
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            output = model(**encoded)

            pooled = mean_pooling(
                output,
                encoded["attention_mask"],
            )

            normalized = torch.nn.functional.normalize(
                pooled,
                p=2,
                dim=1,
            )

            embeddings.append(
                normalized.cpu()
            )

    return torch.cat(embeddings, dim=0)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print("Loading Legal-BERT embedding model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModel.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

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

    hypothesis_lookup = build_hypothesis_lookup(
        raw_data
    )

    gold_by_document = (
        build_gold_evidence_by_document(documents)
    )

    recall_hits = {
        k: 0
        for k in K_VALUES
    }

    reciprocal_ranks = []
    query_results = []
    evaluated_queries = 0
    skipped_missing_clause_alignment = 0

    for document_id, gold_by_hypothesis in (
        gold_by_document.items()
    ):
        document_clauses = clauses_by_document[
            document_id
        ]

        if not document_clauses:
            continue

        print(
            f"Encoding document {document_id} "
            f"({len(document_clauses)} clauses)"
        )

        clause_texts = [
            str(
                clause.get(INDEXED_CLAUSE_FIELD)
                or clause.get("text", "")
            )
            for clause in document_clauses
        ]

        clause_embeddings = encode_texts(
            texts=clause_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        document = raw_document_by_id[document_id]

        for hypothesis_id, gold_span_indices in (
            gold_by_hypothesis.items()
        ):
            hypothesis = hypothesis_lookup.get(
                hypothesis_id,
                "",
            )

            if not hypothesis:
                continue

            gold_ranges = [
                document["spans"][span_index]
                for span_index in gold_span_indices
                if 0 <= span_index
                < len(document["spans"])
            ]

            gold_clause_ids = relevant_clause_ids(
                document_clauses,
                gold_ranges,
            )

            if not gold_clause_ids:
                skipped_missing_clause_alignment += 1
                continue

            query_embedding = encode_texts(
                texts=[hypothesis],
                tokenizer=tokenizer,
                model=model,
                device=device,
            )[0]

            scores = torch.matmul(
                clause_embeddings,
                query_embedding,
            ).numpy()

            ranked_indices = np.argsort(-scores)

            ranked_results = [
                {
                    "clause_id": str(
                        document_clauses[index]["clause_id"]
                    ),
                    "score": float(scores[index]),
                    "text": document_clauses[index].get(
                        "text",
                        "",
                    ),
                }
                for index in ranked_indices
            ]

            ranked_clause_ids = [
                result["clause_id"]
                for result in ranked_results
            ]

            evaluated_queries += 1

            first_relevant_rank = next(
                (
                    rank
                    for rank, clause_id in enumerate(
                        ranked_clause_ids,
                        start=1,
                    )
                    if clause_id in gold_clause_ids
                ),
                None,
            )

            reciprocal_ranks.append(
                1 / first_relevant_rank
                if first_relevant_rank
                else 0.0
            )

            for k in K_VALUES:
                if any(
                    clause_id in gold_clause_ids
                    for clause_id in ranked_clause_ids[:k]
                ):
                    recall_hits[k] += 1

            candidate_scores = {
                str(document_clauses[index]["clause_id"]): float(scores[index])
                for index in range(len(document_clauses))
            }

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
                    "first_relevant_rank":
                        first_relevant_rank,
                    "top_10": ranked_results[:10],
                    "candidate_scores": candidate_scores,
                }
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
    score_results = []

    for query_result in query_results:
        score_results.append(
            {
                "document_id": query_result["document_id"],
                "hypothesis_id": query_result["hypothesis_id"],
                "hypothesis": query_result["hypothesis"],
                "candidates": query_result["top_10"],
                "candidate_scores": query_result["candidate_scores"],
            }
        )
    SCORES_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SCORES_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            score_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved semantic scores: {SCORES_OUTPUT_PATH}"
    )
    output = {
        "configuration": {
            "dataset_split": "dev",
            "retrieval_method":
                "legalbert_mean_pooled_embeddings",
            "model_path": str(MODEL_PATH),
            "indexed_clause_field":
                INDEXED_CLAUSE_FIELD,
            "max_length": MAX_LENGTH,
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

    print("\nLEGAL-BERT SEMANTIC RETRIEVAL")
    print(f"Evaluated queries: {evaluated_queries}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nSaved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()