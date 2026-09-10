# ============================================================
# LEGAL-AWARE TF-IDF RERANKING
# ============================================================

import json
import os
import re
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLAUSE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dev_clauses.json"
)

CONTRACT_NLI_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "contractnli"
    / "contract-nli"
    / "dev.json"
)

SEMANTIC_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval"
    / "dev_legalbert_semantic_scores.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval"
    / "dev_legal_aware_reranking.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_document_id(row):
    return str(
        row.get(
            "document_id",
            row.get("doc_id", "")
        )
    )


def get_clause_id(row):
    return str(
        row.get(
            "clause_id",
            row.get("id", "")
        )
    )


def normalize_text(x):
    return re.sub(
        r"\s+",
        " ",
        str(x).lower()
    ).strip()


def interval_overlaps(
    start_a,
    end_a,
    start_b,
    end_b
):
    return (
        max(start_a, start_b)
        <
        min(end_a, end_b)
    )


# ============================================================
# LOAD PROCESSED CLAUSES
# ============================================================

print("=" * 60)
print("LOADING PROCESSED CLAUSES")
print("=" * 60)

clauses = load_json(CLAUSE_PATH)

df = pd.DataFrame(clauses)

print("Clauses loaded:", len(df))
print()
print("Clause columns:")
print(list(df.columns))

if df.empty:
    raise RuntimeError("Processed clause file is empty.")


# ============================================================
# BUILD CLAUSE INDEX
# ============================================================

print()
print("Unique processed clause IDs:", df["clause_id"].nunique())

clause_lookup = {}

for _, row in df.iterrows():

    clause_id = get_clause_id(row)

    clause_lookup[clause_id] = row


# ============================================================
# LOAD CONTRACTNLI GOLD ANNOTATIONS
# ============================================================

print()
print("=" * 60)
print("LOADING CONTRACTNLI GOLD ANNOTATIONS")
print("=" * 60)

contractnli = load_json(CONTRACT_NLI_PATH)

raw_documents = contractnli.get(
    "documents",
    []
)

print(
    "Raw ContractNLI documents:",
    len(raw_documents)
)


# ============================================================
# BUILD GOLD SPAN MAPPING
# ============================================================

print()
print("Building gold evidence mapping...")

gold_span_mapping = {}

for document in raw_documents:

    document_id = str(
        document["id"]
    )

    annotations = {}

    for annotation_set in document.get(
        "annotation_sets",
        []
    ):

        annotations.update(
            annotation_set.get(
                "annotations",
                {}
            )
        )

    for hypothesis_id, annotation in annotations.items():

        hypothesis_id = str(
            hypothesis_id
        )

        span_ids = annotation.get(
            "spans",
            []
        )

        if span_ids:

            gold_span_mapping[
                (
                    document_id,
                    hypothesis_id
                )
            ] = set(
                int(x)
                for x in span_ids
            )

print(
    "Gold mapping entries:",
    len(gold_span_mapping)
)


# ============================================================
# CONVERT GOLD SPANS TO PROCESSED CLAUSE IDS
# ============================================================

print()
print("Mapping gold spans to processed clauses...")

gold_clause_mapping = {}

clauses_by_document = {}

for index, row in df.iterrows():

    document_id = get_document_id(row)

    clauses_by_document.setdefault(
        document_id,
        []
    ).append(index)


for (
    document_id,
    hypothesis_id
), gold_span_ids in gold_span_mapping.items():

    if document_id not in clauses_by_document:
        continue

    raw_document = None

    for document in raw_documents:

        if str(document["id"]) == document_id:

            raw_document = document
            break

    if raw_document is None:
        continue

    raw_spans = raw_document.get(
        "spans",
        []
    )

    gold_clause_ids = set()

    for span_id in gold_span_ids:

        if span_id < 0:
            continue

        if span_id >= len(raw_spans):
            continue

        evidence_start, evidence_end = (
            raw_spans[span_id]
        )

        for clause_index in clauses_by_document[
            document_id
        ]:

            clause = df.iloc[
                clause_index
            ]

            clause_start = clause.get(
                "start_char"
            )

            clause_end = clause.get(
                "end_char"
            )

            if (
                clause_start is None
                or clause_end is None
            ):
                continue

            try:

                clause_start = int(
                    clause_start
                )

                clause_end = int(
                    clause_end
                )

            except Exception:
                continue

            if interval_overlaps(
                clause_start,
                clause_end,
                int(evidence_start),
                int(evidence_end)
            ):

                gold_clause_ids.add(
                    get_clause_id(
                        clause
                    )
                )

    if gold_clause_ids:

        gold_clause_mapping[
            (
                document_id,
                hypothesis_id
            )
        ] = gold_clause_ids


print(
    "Gold mappings converted to processed clauses:",
    len(gold_clause_mapping)
)

if gold_clause_mapping:

    first_key = next(
        iter(gold_clause_mapping)
    )

    print()
    print("Sample gold mapping:")
    print(
        "Document ID:",
        first_key[0]
    )
    print(
        "Hypothesis ID:",
        first_key[1]
    )
    print(
        "Gold clause IDs:",
        sorted(
            gold_clause_mapping[first_key]
        )
    )


# ============================================================
# LOAD LEGAL-BERT SEMANTIC RESULTS
# ============================================================

print()
print("=" * 60)
print("LOADING LEGAL-BERT SEMANTIC RESULTS")
print("=" * 60)

semantic_results = load_json(
    SEMANTIC_RESULTS_PATH
)

print(
    "Legal-BERT result entries:",
    len(semantic_results)
)

if not semantic_results:

    raise RuntimeError(
        "Legal-BERT semantic result file is empty."
    )


# ============================================================
# BUILD SEMANTIC QUERY LOOKUP
# ============================================================

semantic_lookup = {}

for result in semantic_results:

    if not isinstance(result, dict):
        continue

    document_id = str(
        result.get(
            "document_id",
            ""
        )
    )

    hypothesis_id = str(
        result.get(
            "hypothesis_id",
            ""
        )
    )

    hypothesis = str(
        result.get(
            "hypothesis",
            ""
        )
    )

    candidates = result.get(
        "candidates",
        []
    )

    candidate_scores = result.get(
        "candidate_scores",
        {}
    )

    if (
        not document_id
        or not hypothesis_id
        or not hypothesis
        or not isinstance(candidates, list)
        or not isinstance(candidate_scores, dict)
    ):
        continue

    semantic_lookup[
        (
            document_id,
            hypothesis_id
        )
    ] = {
        "document_id": document_id,
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis,
        "candidates": candidates,
        "candidate_scores": candidate_scores
    }


print(
    "Semantic lookup entries:",
    len(semantic_lookup)
)


# ============================================================
# QUERY PREPARATION
# ============================================================

print()
print("=" * 60)
print("PREPARING LEGAL-AWARE RERANKING QUERIES")
print("=" * 60)

query_records = []

missing_gold = 0
missing_semantic = 0
missing_candidates = 0

for key, gold_ids in gold_clause_mapping.items():

    document_id, hypothesis_id = key

    semantic = semantic_lookup.get(
        key
    )

    if semantic is None:

        missing_semantic += 1
        continue

    candidates = semantic["candidates"]

    if not candidates:

        missing_candidates += 1
        continue

    if not gold_ids:

        missing_gold += 1
        continue

    query_records.append(
        {
            "document_id": document_id,
            "hypothesis_id": hypothesis_id,
            "hypothesis": semantic["hypothesis"],
            "candidates": candidates,
            "candidate_scores": semantic[
                "candidate_scores"
            ],
            "gold_clause_ids": gold_ids
        }
    )


print(
    "Gold mappings available :",
    len(gold_clause_mapping)
)

print(
    "Semantic queries available:",
    len(semantic_lookup)
)

print(
    "Valid queries:",
    len(query_records)
)

print(
    "Missing semantic:",
    missing_semantic
)

print(
    "Missing candidates:",
    missing_candidates
)

print(
    "Missing gold:",
    missing_gold
)

if not query_records:

    raise RuntimeError(
        """
No evaluable legal-aware reranking queries were produced.

Gold evidence and Legal-BERT results exist,
but they could not be joined using:

(document_id, hypothesis_id)
"""
    )


# ============================================================
# BUILD GLOBAL TF-IDF
# ============================================================

print()
print("=" * 60)
print("BUILDING TF-IDF REPRESENTATION")
print("=" * 60)

texts = (
    df["text"]
    .fillna("")
    .astype(str)
    .tolist()
)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(
    texts
)

print(
    "TF-IDF matrix:",
    tfidf_matrix.shape
)


# ============================================================
# LEGAL-AWARE FEATURES
# ============================================================

legal_terms = {
    "shall",
    "must",
    "may",
    "will",
    "prohibited",
    "prohibit",
    "required",
    "require",
    "permitted",
    "permission",
    "terminate",
    "termination",
    "confidential",
    "disclose",
    "disclosure",
    "liable",
    "liability",
    "indemnify",
    "indemnification",
    "consent",
    "notice",
    "agreement",
    "obligation",
    "duty",
    "rights",
    "right",
    "breach",
    "default",
    "payment",
    "license",
    "licence",
    "assignment",
    "restriction"
}


def extract_words(text):

    return set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            normalize_text(text)
        )
    )


def legal_overlap(
    query_words,
    text
):

    words = extract_words(
        text
    )

    query_legal = (
        query_words
        .intersection(
            legal_terms
        )
    )

    if not query_legal:
        return 0.0

    candidate_legal = (
        words
        .intersection(
            legal_terms
        )
    )

    return (
        len(
            query_legal.intersection(
                candidate_legal
            )
        )
        /
        len(query_legal)
    )


def context_overlap(
    query_words,
    context
):

    if not context:
        return 0.0

    context_words = extract_words(
        context
    )

    if not query_words:
        return 0.0

    return (
        len(
            query_words.intersection(
                context_words
            )
        )
        /
        len(query_words)
    )


def modality_match(
    query_text,
    clause
):

    query_words = extract_words(
        query_text
    )

    modality = normalize_text(
        clause.get(
            "modality",
            ""
        )
    )

    if not modality:
        return 0.0

    text = normalize_text(
        clause.get(
            "text",
            ""
        )
    )

    if modality in text:
        return 1.0

    # Also check whether the modality itself
    # occurs in the hypothesis.
    if modality in query_words:
        return 1.0

    return 0.0


# ============================================================
# RERANKING
# ============================================================

print()
print("=" * 60)
print("LEGAL-AWARE TF-IDF RERANKING")
print("=" * 60)


candidate_rows = []

for query in query_records:

    document_id = query[
        "document_id"
    ]

    hypothesis_id = query[
        "hypothesis_id"
    ]

    hypothesis = query[
        "hypothesis"
    ]

    candidates = query[
        "candidates"
    ]

    candidate_scores = query[
        "candidate_scores"
    ]

    gold_ids = set(
        str(x)
        for x in query[
            "gold_clause_ids"
        ]
    )

    query_words = extract_words(
        hypothesis
    )

    # --------------------------------------------------------
    # Resolve candidate IDs
    # --------------------------------------------------------

    candidate_ids = []

    for candidate in candidates:

        if isinstance(candidate, dict):

            candidate_id = (
                candidate.get(
                    "clause_id",
                    candidate.get(
                        "candidate_clause_id",
                        candidate.get(
                            "id",
                            ""
                        )
                    )
                )
            )

        else:

            candidate_id = candidate

        candidate_id = str(
            candidate_id
        )

        if candidate_id in clause_lookup:

            candidate_ids.append(
                candidate_id
            )

    # --------------------------------------------------------
    # If semantic candidates do not directly contain all
    # processed IDs, use candidate_scores keys.
    # --------------------------------------------------------

    if not candidate_ids:

        for candidate_id in candidate_scores.keys():

            candidate_id = str(
                candidate_id
            )

            if candidate_id in clause_lookup:

                candidate_ids.append(
                    candidate_id
                )

    candidate_ids = list(
        dict.fromkeys(
            candidate_ids
        )
    )

    if not candidate_ids:
        continue

    # --------------------------------------------------------
    # Retrieve TF-IDF vectors
    # --------------------------------------------------------

    query_vector = vectorizer.transform(
        [hypothesis]
    )

    candidate_indices = [
        df.index[
            df["clause_id"].astype(str)
            == cid
        ][0]
        for cid in candidate_ids
    ]

    candidate_vectors = (
        tfidf_matrix[
            candidate_indices
        ]
    )

    tfidf_scores = cosine_similarity(
        query_vector,
        candidate_vectors
    )[0]

    # --------------------------------------------------------
    # Normalize semantic scores
    # --------------------------------------------------------

    semantic_values = []

    for cid in candidate_ids:

        value = candidate_scores.get(
            cid,
            candidate_scores.get(
                str(cid),
                0.0
            )
        )

        try:
            value = float(
                value
            )
        except Exception:
            value = 0.0

        semantic_values.append(
            value
        )

    semantic_values = np.array(
        semantic_values,
        dtype=float
    )

    # Min-max normalization
    if (
        semantic_values.max()
        >
        semantic_values.min()
    ):

        semantic_norm = (
            semantic_values
            -
            semantic_values.min()
        ) / (
            semantic_values.max()
            -
            semantic_values.min()
        )

    else:

        semantic_norm = np.zeros_like(
            semantic_values
        )

    # --------------------------------------------------------
    # Candidate feature calculation
    # --------------------------------------------------------

    rows = []

    for i, clause_id in enumerate(
        candidate_ids
    ):

        clause = clause_lookup[
            clause_id
        ]

        text = str(
            clause.get(
                "text",
                ""
            )
        )

        context = str(
            clause.get(
                "context_text",
                ""
            )
        )

        legal_score = legal_overlap(
            query_words,
            text
        )

        modality_score = modality_match(
            hypothesis,
            clause
        )

        context_score = context_overlap(
            query_words,
            context
        )

        rows.append(
            {
                "document_id": document_id,
                "hypothesis_id": hypothesis_id,
                "hypothesis": hypothesis,
                "candidate_clause_id": clause_id,
                "semantic_score": float(
                    semantic_values[i]
                ),
                "semantic_normalized": float(
                    semantic_norm[i]
                ),
                "tfidf_score": float(
                    tfidf_scores[i]
                ),
                "legal_overlap": float(
                    legal_score
                ),
                "modality_match": float(
                    modality_score
                ),
                "context_overlap": float(
                    context_score
                ),
                "is_gold": (
                    clause_id
                    in gold_ids
                )
            }
        )

    if not rows:
        continue

    candidate_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Legal-aware score
    #
    # We combine:
    #   TF-IDF semantic lexical similarity
    #   Legal term overlap
    #   Modality compatibility
    #   Context overlap
    # --------------------------------------------------------

    candidate_df[
        "rerank_score"
    ] = (
        0.70
        *
        candidate_df[
            "tfidf_score"
        ]
        +
        0.15
        *
        candidate_df[
            "legal_overlap"
        ]
        +
        0.10
        *
        candidate_df[
            "modality_match"
        ]
        +
        0.05
        *
        candidate_df[
            "context_overlap"
        ]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidate_df = (
        candidate_df
        .sort_values(
            "rerank_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    candidate_df[
        "rank"
    ] = (
        np.arange(
            len(candidate_df)
        )
        +
        1
    )

    candidate_rows.extend(
        candidate_df.to_dict(
            orient="records"
        )
    )


results_df = pd.DataFrame(
    candidate_rows
)

print()
print(
    "Evaluated queries:",
    results_df[
        [
            "document_id",
            "hypothesis_id"
        ]
    ].drop_duplicates().shape[0]
    if not results_df.empty
    else 0
)

print(
    "Candidate rows:",
    len(results_df)
)


if results_df.empty:

    raise RuntimeError(
        """
No evaluable legal-aware reranking queries were produced.

Check the candidate clause IDs in the Legal-BERT
semantic result file against processed clause_id.
"""
    )


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    df,
    k
):

    grouped = df.groupby(
        [
            "document_id",
            "hypothesis_id"
        ]
    )

    hits = 0
    total = 0

    for _, group in grouped:

        total += 1

        top_k = group[
            group["rank"] <= k
        ]

        if top_k[
            "is_gold"
        ].any():

            hits += 1

    return (
        hits / total
        if total
        else 0.0
    )


# ============================================================
# MRR
# ============================================================

def mean_reciprocal_rank(
    df
):

    grouped = df.groupby(
        [
            "document_id",
            "hypothesis_id"
        ]
    )

    reciprocal_ranks = []

    for _, group in grouped:

        gold = group[
            group["is_gold"]
        ]

        if gold.empty:

            reciprocal_ranks.append(
                0.0
            )

        else:

            first_rank = gold[
                "rank"
            ].min()

            reciprocal_ranks.append(
                1.0 / first_rank
            )

    return (
        float(
            np.mean(
                reciprocal_ranks
            )
        )
        if reciprocal_ranks
        else 0.0
    )


# ============================================================
# FINAL METRICS
# ============================================================

print()
print("=" * 60)
print("FINAL LEGAL-AWARE RERANKING RESULTS")
print("=" * 60)

metrics = {}

for k in [
    1,
    2,
    5,
    10
]:

    value = recall_at_k(
        results_df,
        k
    )

    metrics[
        f"recall_at_{k}"
    ] = value

    print(
        f"Recall@{k}: {value:.4f}"
    )


mrr = mean_reciprocal_rank(
    results_df
)

metrics[
    "mrr"
] = mrr

print(
    f"MRR: {mrr:.4f}"
)


# ============================================================
# SAVE REPORT
# ============================================================

os.makedirs(
    OUTPUT_PATH.parent,
    exist_ok=True
)

report = {
    "method":
        "legal_aware_tfidf_reranking",

    "queries":
        int(
            results_df[
                [
                    "document_id",
                    "hypothesis_id"
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),

    "candidate_rows":
        int(
            len(results_df)
        ),

    "metrics":
        metrics
}

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

DETAIL_PATH = (
    OUTPUT_PATH.parent
    /
    "dev_legal_aware_reranking_details.json"
)

details = results_df.to_dict(
    orient="records"
)

with open(
    DETAIL_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        details,
        f,
        indent=2
    )


print()
print("Saved report:")
print(
    OUTPUT_PATH.resolve()
)

print()
print("Saved detailed results:")
print(
    DETAIL_PATH.resolve()
)