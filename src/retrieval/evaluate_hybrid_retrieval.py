import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLAUSES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dev_clauses.json"
)

RAW_CONTRACTNLI_PATH = (
    PROJECT_ROOT
    / "data"
    / "kaggle"
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
    / "dev_hybrid_retrieval.json"
)

TOP_K_VALUES = [1, 2, 5, 10]

ALPHAS = [0.25, 0.50, 0.75]


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


print("=" * 60)
print("LOADING PROCESSED CLAUSES")
print("=" * 60)

clauses = load_json(CLAUSES_PATH)

print("Loaded clauses:", len(clauses))


# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame(clauses)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# BASIC HELPERS
# ============================================================

def get_text(clause):
    """
    Text used for TF-IDF.

    Prefer context_text if available, otherwise use text.
    """
    return str(
        clause.get(
            "context_text",
            clause.get("text", "")
        )
    )


def get_clause_id(clause):
    return str(clause.get("clause_id"))


def get_document_id(clause):
    return str(clause.get("document_id"))


# ============================================================
# BUILD CLAUSE INDEX
# ============================================================

print("\nBuilding clause index...")

clause_id_to_index = {}

for i in range(len(df)):

    clause_id = get_clause_id(df.iloc[i])

    clause_id_to_index[clause_id] = i


print(
    "Unique processed clause IDs:",
    len(clause_id_to_index)
)


# ============================================================
# TF-IDF REPRESENTATION
# ============================================================

print("\nBuilding TF-IDF representation...")

texts = [
    get_text(df.iloc[i])
    for i in range(len(df))
]

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(texts)

print(
    "TF-IDF matrix:",
    X_tfidf.shape
)


# ============================================================
# LOAD CONTRACTNLI RAW DATA
# ============================================================

print("\nLoading ContractNLI gold annotations...")

raw_data = load_json(RAW_CONTRACTNLI_PATH)

raw_documents = raw_data["documents"]
raw_labels = raw_data["labels"]

print(
    "Raw ContractNLI documents:",
    len(raw_documents)
)

print(
    "ContractNLI hypotheses:",
    len(raw_labels)
)


# ============================================================
# BUILD GOLD EVIDENCE MAPPING
#
# Mapping:
#
# (document_id, hypothesis_id)
#          ->
# set of ContractNLI span IDs
#
# ============================================================

print("\nBuilding gold evidence mapping...")


gold_span_mapping = {}

for document in raw_documents:

    document_id = str(
        document["id"]
    )

    annotations = {}

    # ContractNLI normally has one annotation set,
    # but this safely handles multiple annotation sets.

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

        # Store only examples that have evidence.
        # ContractNLI NotMentioned examples have
        # empty evidence spans.

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
# CONVERT CONTRACTNLI SPAN IDs TO PROCESSED CLAUSE IDs
#
# IMPORTANT:
#
# ContractNLI's gold spans are integer span IDs.
#
# Your processed clauses contain start_char/end_char.
#
# Therefore we match:
#
# raw document span ID
#        ↓
# raw character range
#        ↓
# processed clause character range
#
# using interval overlap.
#
# ============================================================

def interval_overlaps(
    start_a,
    end_a,
    start_b,
    end_b
):
    return (
        max(
            start_a,
            start_b
        )
        <
        min(
            end_a,
            end_b
        )
    )


print(
    "\nMapping gold spans to processed clauses..."
)


gold_clause_mapping = {}


# Group processed clauses by document
clauses_by_document = {}

for i in range(len(df)):

    document_id = get_document_id(
        df.iloc[i]
    )

    clauses_by_document.setdefault(
        document_id,
        []
    ).append(i)


for (
    document_id,
    hypothesis_id
), gold_span_ids in gold_span_mapping.items():

    if document_id not in clauses_by_document:
        continue

    # Find raw document
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

    # Convert every gold span ID into
    # its character interval.

    for span_id in gold_span_ids:

        if span_id < 0:
            continue

        if span_id >= len(raw_spans):
            continue

        evidence_start, evidence_end = raw_spans[
            span_id
        ]

        # Compare with processed clauses
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

            except:

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


# ============================================================
# SHOW SAMPLE GOLD MAPPING
# ============================================================

if gold_clause_mapping:

    first_key = next(
        iter(gold_clause_mapping)
    )

    print("\nSample gold mapping:")

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
        list(
            gold_clause_mapping[first_key]
        )[:10]
    )


# ============================================================
# LOAD LEGAL-BERT RESULTS
# ============================================================

print("\nLoading Legal-BERT semantic results...")

semantic_results = load_json(
    SEMANTIC_RESULTS_PATH
)

print(
    "Legal-BERT result entries:",
    len(semantic_results)
)


# ============================================================
# CHECK SEMANTIC RESULT STRUCTURE
# ============================================================

if semantic_results:

    first_result = semantic_results[0]

    print("\nFirst semantic result fields:")

    for key, value in first_result.items():

        if isinstance(value, list):

            print(
                f"  {key}: list ({len(value)} entries)"
            )

        elif isinstance(value, dict):

            print(
                f"  {key}: dictionary ({len(value)} entries)"
            )

        else:

            print(
                f"  {key}: {type(value).__name__}"
            )


# ============================================================
# INDEX SEMANTIC RESULTS
#
# Key:
# (document_id, hypothesis_id)
#
# ============================================================

semantic_lookup = {}

for result in semantic_results:

    document_id = str(
        result.get(
            "document_id"
        )
    )

    hypothesis_id = str(
        result.get(
            "hypothesis_id"
        )
    )

    semantic_lookup[
        (
            document_id,
            hypothesis_id
        )
    ] = result


print(
    "\nSemantic lookup entries:",
    len(semantic_lookup)
)


# ============================================================
# PREPARE HYBRID QUERIES
# ============================================================

print("\n" + "=" * 60)
print("HYBRID RETRIEVAL QUERY PREPARATION")
print("=" * 60)


queries = []

missing_gold = 0
missing_candidates = 0
missing_semantic_scores = 0
missing_document = 0
missing_clause_ids = 0


for semantic_result in semantic_results:

    document_id = str(
        semantic_result.get(
            "document_id"
        )
    )

    hypothesis_id = str(
        semantic_result.get(
            "hypothesis_id"
        )
    )

    key = (
        document_id,
        hypothesis_id
    )

    # --------------------------------------------------------
    # GOLD
    # --------------------------------------------------------

    gold = gold_clause_mapping.get(
        key,
        set()
    )

    if not gold:

        missing_gold += 1
        continue

    # --------------------------------------------------------
    # SEMANTIC CANDIDATES
    # --------------------------------------------------------

    semantic_candidates = (
        semantic_result.get(
            "candidates",
            []
        )
    )

    if not semantic_candidates:

        missing_candidates += 1
        continue

    # --------------------------------------------------------
    # SEMANTIC SCORES
    # --------------------------------------------------------

    semantic_scores_dict = (
        semantic_result.get(
            "candidate_scores",
            {}
        )
    )

    if not semantic_scores_dict:

        missing_semantic_scores += 1
        continue

    # --------------------------------------------------------
    # FIND PROCESSED CLAUSES FOR CANDIDATES
    # --------------------------------------------------------

    candidate_indices = []

    for candidate in semantic_candidates:

        candidate_clause_id = str(
            candidate.get(
                "clause_id"
            )
        )

        if candidate_clause_id not in clause_id_to_index:

            missing_clause_ids += 1
            continue

        candidate_index = (
            clause_id_to_index[
                candidate_clause_id
            ]
        )

        candidate_document_id = (
            get_document_id(
                df.iloc[
                    candidate_index
                ]
            )
        )

        # Make sure candidate belongs
        # to the same document.

        if candidate_document_id != document_id:

            continue

        candidate_indices.append(
            candidate_index
        )

    if not candidate_indices:

        missing_candidates += 1
        continue

    # --------------------------------------------------------
    # QUERY TEXT
    # --------------------------------------------------------

    hypothesis = str(
        semantic_result.get(
            "hypothesis",
            ""
        )
    )

    queries.append(
        {
            "document_id": document_id,
            "hypothesis_id": hypothesis_id,
            "hypothesis": hypothesis,
            "candidate_indices": candidate_indices,
            "gold": gold
        }
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

print(
    "\nSemantic result entries :",
    len(semantic_results)
)

print(
    "Gold mappings available  :",
    len(gold_clause_mapping)
)

print(
    "Valid queries           :",
    len(queries)
)

print(
    "Missing gold            :",
    missing_gold
)

print(
    "Missing candidates      :",
    missing_candidates
)

print(
    "Missing semantic scores:",
    missing_semantic_scores
)

print(
    "Missing clause IDs     :",
    missing_clause_ids
)


# ============================================================
# FAIL ONLY IF SOMETHING IS ACTUALLY BROKEN
# ============================================================

if len(queries) == 0:

    raise RuntimeError(
        "\nNo evaluable queries were constructed.\n\n"
        "Check the diagnostics above.\n"
        "The expected pipeline is:\n"
        "ContractNLI dev.json -> gold spans -> "
        "processed clause IDs -> Legal-BERT candidates -> "
        "hybrid evaluation."
    )


# ============================================================
# VERIFY GOLD IS ACTUALLY PRESENT IN CANDIDATES
# ============================================================

print(
    "\nChecking candidate coverage..."
)


coverage_hits = 0

for q in queries:

    candidate_ids = set(
        get_clause_id(
            df.iloc[index]
        )
        for index in q[
            "candidate_indices"
        ]
    )

    if candidate_ids.intersection(
        q["gold"]
    ):

        coverage_hits += 1


coverage = (
    coverage_hits / len(queries)
    if queries
    else 0
)


print(
    "Queries:",
    len(queries)
)

print(
    "Queries with gold in candidates:",
    coverage_hits
)

print(
    "Candidate gold coverage:",
    f"{coverage:.4f}",
    f"({coverage * 100:.2f}%)"
)


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:
        return values

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        return np.zeros_like(
            values
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def recall_at_k(
    rankings,
    gold,
    k
):

    if not gold:
        return None

    retrieved = set(
        rankings[:k]
    )

    return (
        1.0
        if retrieved.intersection(
            gold
        )
        else 0.0
    )


def reciprocal_rank(
    rankings,
    gold
):

    if not gold:
        return None

    for rank, clause_id in enumerate(
        rankings,
        start=1
    ):

        if clause_id in gold:

            return 1.0 / rank

    return 0.0


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

all_results = []


for alpha in ALPHAS:

    print(
        "\n" + "=" * 60
    )

    print(
        "ALPHA =",
        alpha
    )

    print(
        "=" * 60
    )

    metric_values = {
        k: []
        for k in TOP_K_VALUES
    }

    mrr_values = []

    evaluated = 0

    # --------------------------------------------------------
    # PROCESS EACH QUERY
    # --------------------------------------------------------

    for q in queries:

        document_id = q[
            "document_id"
        ]

        hypothesis_id = q[
            "hypothesis_id"
        ]

        candidate_indices = q[
            "candidate_indices"
        ]

        gold = q[
            "gold"
        ]

        semantic_result = semantic_lookup.get(
            (
                document_id,
                hypothesis_id
            )
        )

        if semantic_result is None:

            continue

        semantic_scores_dict = (
            semantic_result.get(
                "candidate_scores",
                {}
            )
        )

        # ----------------------------------------------------
        # TF-IDF QUERY
        #
        # Use the hypothesis as the query vector.
        # ----------------------------------------------------

        query_text = q[
            "hypothesis"
        ]

        query_vector = (
            vectorizer.transform(
                [query_text]
            )
        )

        tfidf_scores = cosine_similarity(
            query_vector,
            X_tfidf[
                candidate_indices
            ]
        )[0]

        # ----------------------------------------------------
        # LEGAL-BERT SCORES
        # ----------------------------------------------------

        semantic_scores = np.array(
            [
                float(
                    semantic_scores_dict.get(
                        get_clause_id(
                            df.iloc[j]
                        ),
                        0.0
                    )
                )
                for j in candidate_indices
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        tfidf_norm = (
            minmax_normalize(
                tfidf_scores
            )
        )

        semantic_norm = (
            minmax_normalize(
                semantic_scores
            )
        )

        # ----------------------------------------------------
        # HYBRID SCORE
        #
        # alpha controls TF-IDF contribution.
        #
        # alpha = 0.25:
        #     25% TF-IDF
        #     75% Legal-BERT
        #
        # alpha = 0.50:
        #     50% TF-IDF
        #     50% Legal-BERT
        #
        # alpha = 0.75:
        #     75% TF-IDF
        #     25% Legal-BERT
        # ----------------------------------------------------

        hybrid_scores = (
            alpha * tfidf_norm
            + (
                1.0 - alpha
            ) * semantic_norm
        )

        # ----------------------------------------------------
        # RANK
        # ----------------------------------------------------

        ranking_order = np.argsort(
            -hybrid_scores
        )

        ranked_ids = [
            get_clause_id(
                df.iloc[
                    candidate_indices[idx]
                ]
            )
            for idx in ranking_order
        ]

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        evaluated += 1

        for k in TOP_K_VALUES:

            value = recall_at_k(
                ranked_ids,
                gold,
                k
            )

            if value is not None:

                metric_values[
                    k
                ].append(
                    value
                )

        rr = reciprocal_rank(
            ranked_ids,
            gold
        )

        if rr is not None:

            mrr_values.append(
                rr
            )

    # ========================================================
    # RESULTS FOR THIS ALPHA
    # ========================================================

    row = {
        "alpha": alpha,
        "evaluated_queries": evaluated
    }

    for k in TOP_K_VALUES:

        if metric_values[k]:

            score = np.mean(
                metric_values[k]
            )

        else:

            score = 0.0

        row[
            f"recall_at_{k}"
        ] = score

    if mrr_values:

        row["mrr"] = np.mean(
            mrr_values
        )

    else:

        row["mrr"] = 0.0

    all_results.append(
        row
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Evaluated queries: {evaluated}"
    )

    print(
        f"Recall@1  : "
        f"{row['recall_at_1']:.4f}"
    )

    print(
        f"Recall@2  : "
        f"{row['recall_at_2']:.4f}"
    )

    print(
        f"Recall@5  : "
        f"{row['recall_at_5']:.4f}"
    )

    print(
        f"Recall@10 : "
        f"{row['recall_at_10']:.4f}"
    )

    print(
        f"MRR       : "
        f"{row['mrr']:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print(
    "\n" + "=" * 60
)

print(
    "FINAL HYBRID RETRIEVAL RESULTS"
)

print(
    "=" * 60
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=2
    )


print(
    "\nSaved:",
    OUTPUT_PATH
)