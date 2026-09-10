import json
from pathlib import Path

from src.retrieval.candidate_retrieval import CandidateRetriever
from src.reasoning.conflict_engine import ConflictEngine
from src.reasoning.hybrid_reasoner import HybridReasoner
from src.nli.legalbert_inference import LegalBERTInference

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "models/lexconflict_legalbert"

DATA_PATH = Path("data/processed/test_clauses.json")

OUTPUT_PATH = Path(
    "results/reasoning/conflict_detection.json"
)

TOP_K = 5


# ============================================================
# LOAD CLAUSES
# ============================================================

def load_clauses(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("LEXCONFLICT - FULL CONFLICT DETECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load processed clauses
    # --------------------------------------------------------

    print("\nLoading processed clauses...")

    clauses = load_clauses(DATA_PATH)

    print(f"Clauses loaded: {len(clauses)}")

    # --------------------------------------------------------
    # 2. Initialize retrieval
    # --------------------------------------------------------

    print("\nFitting candidate retrieval model...")

    retriever = CandidateRetriever(
        top_k=TOP_K,
        same_document_only=True
    )

    retriever.fit(clauses)

    print("Candidate retrieval ready.")

    # --------------------------------------------------------
    # 3. Initialize reasoning components
    # --------------------------------------------------------

    conflict_engine = ConflictEngine(
        conflict_threshold=0.50
    )

    hybrid_reasoner = HybridReasoner(
        nli_weight=0.95,
        structured_weight=0.05,
        conflict_threshold=0.71
    )
    print("\nLoading Legal-BERT...")
    nli_model = LegalBERTInference(
        MODEL_PATH
    )
    # --------------------------------------------------------
    # 4. Process every clause
    # --------------------------------------------------------

    all_results = []

    total_pairs = 0
    conflicts = 0

    print("\nRunning conflict detection...")

    for index, query_clause in enumerate(clauses):

        if index % 100 == 0:
            print(
                f"Processing clause "
                f"{index + 1}/{len(clauses)}"
            )

        candidates = retriever.retrieve(
            query_clause
        )

        for candidate in candidates:

            candidate_clause = next(
                (
                    clause
                    for clause in clauses
                    if clause.get("clause_id")
                    == candidate.candidate_clause_id
                ),
                None
            )

            if candidate_clause is None:
                continue

            # ------------------------------------------------
            # Structured reasoning
            # ------------------------------------------------

            structured_result = conflict_engine.analyze_pair(
                query_clause,
                candidate_clause,
                semantic_similarity=candidate.score
            )

            # ------------------------------------------------
            # Temporary NLI result
            #
            # IMPORTANT:
            # Replace this section with the actual
            # Legal-BERT inference function from your project.
            # ------------------------------------------------
            # ------------------------------------------------
            # Legal-BERT NLI inference
            # ------------------------------------------------

            nli_result = nli_model.predict(
                query_clause.get("text", ""),
                candidate_clause.get("text", "")
            )

            # ------------------------------------------------
            # Hybrid reasoning
            # ------------------------------------------------

            hybrid_result = hybrid_reasoner.analyze(
                nli_result,
                structured_result
            )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            result = {
                "query_clause_id":
                    query_clause.get("clause_id"),

                "candidate_clause_id":
                    candidate_clause.get("clause_id"),

                "retrieval_score":
                    candidate.score,

                "retrieval_rank":
                    candidate.rank,

                "structured_conflict_score":
                    structured_result.confidence,

                "structured_conflict_type":
                    structured_result.conflict_type,

                "semantic_similarity":
                    structured_result.semantic_similarity,

                "modality_conflict":
                    structured_result.modality_conflict,

                "negation_conflict":
                    structured_result.negation_conflict,

                "same_party":
                    structured_result.same_party,

                "condition_similarity":
                    structured_result.condition_similarity,

                "nli_label":
                    hybrid_result["nli_label"],

                "nli_contradiction_probability":
                    hybrid_result[
                        "nli_contradiction_probability"
                    ],

                "hybrid_conflict_score":
                    hybrid_result[
                        "hybrid_conflict_score"
                    ],

                "is_conflict":
                    hybrid_result["is_conflict"],

                "confidence_level":
                    hybrid_result["confidence_level"],

                "explanation":
                    hybrid_result["explanation"]
            }

            all_results.append(result)

            total_pairs += 1

            if hybrid_result["is_conflict"]:
                conflicts += 1

    # --------------------------------------------------------
    # 5. Save results
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFLICT DETECTION COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal clauses: {len(clauses)}"
    )

    print(
        f"Candidate pairs analyzed: {total_pairs}"
    )

    print(
        f"Potential conflicts: {conflicts}"
    )

    if total_pairs > 0:

        print(
            f"Conflict rate: "
            f"{conflicts / total_pairs:.4f}"
        )

    print(
        f"\nSaved results to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()