import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.models.nli_inference import LegalBERTNLI
from src.retrieval.candidate_retrieval import CandidateRetriever
from src.reasoning.conflict_engine import ConflictEngine
from src.reasoning.hybrid_reasoner import HybridReasoner

from src.graph.graph_node import GraphNode
from src.graph.conflict_graph import ConflictGraph


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = "models/lexconflict_legalbert"

CLAUSE_PATH = (
    "data/processed/dev_clauses.json"
)

OUTPUT_DIR = Path(
    "data/processed/conflict_graph"
)

OUTPUT_GRAPH = (
    OUTPUT_DIR / "dev_conflict_graph.json"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "dev_conflict_graph_summary.json"
)

# Start small because Legal-BERT is running locally on CPU.
MAX_DOCUMENTS = 3

TOP_K = 5

BATCH_SIZE = 16

MAX_LENGTH = 120


# =========================================================
# Load clauses
# =========================================================

def load_clauses(
    path: str
) -> List[Dict]:

    print(
        f"\nLoading clauses from: {path}"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        clauses = json.load(f)

    print(
        f"Clauses loaded: {len(clauses)}"
    )

    return clauses


# =========================================================
# Group clauses by document
# =========================================================

def group_by_document(
    clauses: List[Dict]
) -> Dict[str, List[Dict]]:

    documents = {}

    for clause in clauses:

        document_id = str(
            clause["document_id"]
        )

        documents.setdefault(
            document_id,
            []
        ).append(clause)

    return documents


# =========================================================
# Convert clause to GraphNode
# =========================================================

def clause_to_node(
    clause: Dict
) -> GraphNode:

    clause_id = str(
        clause["clause_id"]
    )

    return GraphNode(
        node_id=clause_id,

        document_id=str(
            clause["document_id"]
        ),

        clause_id=clause_id,

        text=str(
            clause.get("text", "")
        ),

        party=clause.get(
            "party"
        ),

        modality=clause.get(
            "modality"
        ),

        condition=clause.get(
            "condition"
        ),

        proposition=clause.get(
            "proposition"
        ),

        dataset=clause.get(
            "dataset"
        )
    )


# =========================================================
# Convert retrieval result to NLI pair
# =========================================================

def retrieval_to_pair(
    result
) -> Dict:

    return {
        "hypothesis": result.query_text,

        "clause_text": result.candidate_text,

        "clause_id": result.candidate_clause_id,

        "document_id":
            result.query_document_id,

        "query_clause_id":
            result.query_clause_id,

        "retrieval_rank":
            result.rank,

        "retrieval_score":
            result.score
    }


# =========================================================
# Main graph construction
# =========================================================

def build_graph():

    print("=" * 70)
    print("LEXCONFLICT ACTUAL CONFLICT GRAPH")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load clauses
    # -----------------------------------------------------

    all_clauses = load_clauses(
        CLAUSE_PATH
    )

    documents = group_by_document(
        all_clauses
    )

    print(
        f"Documents available: "
        f"{len(documents)}"
    )

    # -----------------------------------------------------
    # 2. Select documents
    # -----------------------------------------------------

    selected_document_ids = list(
        documents.keys()
    )[:MAX_DOCUMENTS]

    print(
        "\nSelected documents:"
    )

    for document_id in selected_document_ids:

        print(
            f"  {document_id}: "
            f"{len(documents[document_id])} clauses"
        )

    selected_clauses = []

    for document_id in selected_document_ids:

        selected_clauses.extend(
            documents[document_id]
        )
        
        clause_lookup = {
            str(clause["clause_id"]): clause
            for clause in selected_clauses
        }

    # -----------------------------------------------------
    # 3. Create graph
    # -----------------------------------------------------

    graph = ConflictGraph()

    for clause in selected_clauses:

        node = clause_to_node(
            clause
        )

        graph.add_node(node)

    print(
        f"\nGraph nodes created: "
        f"{graph.number_of_nodes()}"
    )

    # -----------------------------------------------------
    # 4. Candidate retrieval
    # -----------------------------------------------------

    print(
        "\nBuilding candidate retriever..."
    )

    retriever = CandidateRetriever(
        top_k=TOP_K,
        same_document_only=True
    )

    retriever.fit(
        selected_clauses
    )

    print(
        "Candidate retrieval complete."
    )

    # -----------------------------------------------------
    # 5. Retrieve candidates
    # -----------------------------------------------------

    print(
        "\nRetrieving candidate pairs..."
    )

    retrieval_results = (
        retriever.retrieve_all()
    )

    print(
        f"Retrieved pairs: "
        f"{len(retrieval_results)}"
    )

    # -----------------------------------------------------
    # 6. Remove duplicate pairs
    # -----------------------------------------------------

    unique_results = []

    seen_pairs: Set[
        Tuple[str, str]
    ] = set()

    for result in retrieval_results:

        source = str(
            result.query_clause_id
        )

        target = str(
            result.candidate_clause_id
        )

        # Treat A-B and B-A as the same graph edge.
        pair_key = tuple(
            sorted(
                [source, target]
            )
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(
            pair_key
        )

        unique_results.append(
            result
        )

    print(
        f"Unique candidate pairs: "
        f"{len(unique_results)}"
    )

    # -----------------------------------------------------
    # 7. Create NLI pairs
    # -----------------------------------------------------

    nli_pairs = [
        retrieval_to_pair(result)
        for result in unique_results
    ]

    # -----------------------------------------------------
    # 8. Load Legal-BERT
    # -----------------------------------------------------

    print(
        "\nLoading Legal-BERT..."
    )

    nli_model = LegalBERTNLI(
        model_path=MODEL_PATH,
        device="cpu"
    )

    # -----------------------------------------------------
    # 9. Run NLI
    # -----------------------------------------------------

    print(
        "\nRunning Legal-BERT inference..."
    )

    nli_predictions = nli_model.predict(
        nli_pairs,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        num_workers=0
    )

    print(
        f"NLI predictions: "
        f"{len(nli_predictions)}"
    )

    # -----------------------------------------------------
    # 10. Structured + Hybrid reasoning
    # -----------------------------------------------------

    conflict_engine = ConflictEngine()

    hybrid_reasoner = HybridReasoner(
        nli_weight=0.95,
        structured_weight=0.05,
        conflict_threshold=0.71
    )

    conflict_count = 0
    related_count = 0

    # -----------------------------------------------------
    # 11. Process each candidate pair
    # -----------------------------------------------------

    for index, (
        result,
        nli_prediction
    ) in enumerate(
        zip(
            unique_results,
            nli_predictions
        )
    ):

        

        source_clause = clause_lookup[
            str(result.query_clause_id)
        ]

        target_clause = clause_lookup[
            str(result.candidate_clause_id)
        ]

        # ---------------------------------------------
        # Structured reasoning
        # ---------------------------------------------

        structured_result = (
            conflict_engine.analyze_pair(
                source_clause,
                target_clause
            )
        )

        # ---------------------------------------------
        # Hybrid reasoning
        # ---------------------------------------------

        hybrid_result = (
            hybrid_reasoner.analyze(
                nli_result=nli_prediction,
                structured_result=structured_result
            )
        )

        # ---------------------------------------------
        # Add graph edge
        # ---------------------------------------------

        edge = graph.add_hybrid_result(
            source_id=str(
                result.query_clause_id
            ),

            target_id=str(
                result.candidate_clause_id
            ),

            hybrid_result=hybrid_result
        )

        if edge.relationship == "conflict":

            conflict_count += 1

        else:

            related_count += 1

        # ---------------------------------------------
        # Progress
        # ---------------------------------------------

        if (
            index + 1
        ) % 50 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(unique_results)}"
            )

    # -----------------------------------------------------
    # 12. Create summary
    # -----------------------------------------------------

    summary = graph.summary()

    summary.update({

        "documents_processed":
            len(selected_document_ids),

        "candidate_pairs":
            len(unique_results),

        "conflict_edges":
            conflict_count,

        "related_edges":
            related_count,

        "top_k":
            TOP_K,

        "nli_weight":
            0.95,

        "structured_weight":
            0.05,

        "conflict_threshold":
            0.71
    })

    # -----------------------------------------------------
    # 13. Create output directory
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 14. Save graph
    # -----------------------------------------------------

    graph_data = {

        "nodes": [
            node.__dict__
            for node in graph.get_nodes()
        ],

        "edges": [
            edge.__dict__
            for edge in graph.get_edges()
        ]
    }

    with open(
        OUTPUT_GRAPH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            graph_data,
            f,
            indent=2
        )

    # -----------------------------------------------------
    # 15. Save summary
    # -----------------------------------------------------

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # -----------------------------------------------------
    # 16. Print final result
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFLICT GRAPH COMPLETE")
    print("=" * 70)

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print(
        f"\nGraph saved to:"
        f"\n{OUTPUT_GRAPH}"
    )

    print(
        f"\nSummary saved to:"
        f"\n{OUTPUT_SUMMARY}"
    )


if __name__ == "__main__":

    build_graph()