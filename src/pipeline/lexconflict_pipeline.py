import json
from pathlib import Path
from typing import Dict, List, Any

from src.retrieval.candidate_retrieval import CandidateRetriever
from src.models.nli_inference import LegalBERTNLI
from src.reasoning.conflict_engine import ConflictEngine
from src.reasoning.hybrid_reasoner import HybridReasoner

from src.graph.graph_node import GraphNode
from src.graph.graph_edge import GraphEdge
from src.graph.conflict_graph import ConflictGraph
from src.graph.graph_propagation import GraphPropagation

from src.risk.risk_engine import RiskEngine


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/lexconflict_legalbert"

CLAUSE_PATH = (
    "data/processed/dev_clauses.json"
)

OUTPUT_DIR = Path(
    "data/processed/pipeline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "dev_pipeline_results.json"
)


# ------------------------------------------------------------
# Processing limits
# ------------------------------------------------------------

# Keep this small for the first end-to-end test.
# Once the pipeline works, we can increase it.
MAX_DOCUMENTS = 1

TOP_K = 2

NLI_BATCH_SIZE = 16

NLI_MAX_LENGTH = 120


# ------------------------------------------------------------
# Hybrid configuration
# Selected using DEV tuning
# ------------------------------------------------------------

NLI_WEIGHT = 0.95

STRUCTURED_WEIGHT = 0.05

CONFLICT_THRESHOLD = 0.71


# ------------------------------------------------------------
# Graph propagation configuration
# ------------------------------------------------------------

PROPAGATION_ALPHA = 0.70

PROPAGATION_ITERATIONS = 3


# ------------------------------------------------------------
# Risk configuration
# ------------------------------------------------------------

PROPAGATED_WEIGHT = 0.50

DIRECT_WEIGHT = 0.30

CONNECTIVITY_WEIGHT = 0.20


# ============================================================
# LOAD CLAUSES
# ============================================================

def load_clauses(
    path: str
) -> List[Dict[str, Any]]:

    print(
        f"\nLoading clauses from: {path}"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        clauses = json.load(file)

    print(
        f"Clauses loaded: {len(clauses)}"
    )

    return clauses


# ============================================================
# GROUP CLAUSES BY DOCUMENT
# ============================================================

def group_by_document(
    clauses: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:

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


# ============================================================
# CONVERT CLAUSE TO GRAPH NODE
# ============================================================

def clause_to_node(
    clause: Dict[str, Any]
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


# ============================================================
# CONVERT RETRIEVAL RESULT TO NLI PAIR
# ============================================================

def retrieval_to_pair(
    result,
    clause_lookup: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convert a retrieval result into an NLI pair.

    If a clause has parent context, use the context-aware
    representation for NLI. Otherwise, use the original text.
    """

    query_clause = clause_lookup.get(
        str(result.query_clause_id),
        {}
    )

    candidate_clause = clause_lookup.get(
        str(result.candidate_clause_id),
        {}
    )

    query_context = query_clause.get(
        "context_text"
    ) or result.query_text

    candidate_context = candidate_clause.get(
        "context_text"
    ) or result.candidate_text

    return {
        "hypothesis": query_context,

        "evidence": candidate_context,

        "clause_text":
            result.candidate_text,

        "clause_id":
            result.candidate_clause_id,

        "document_id":
            result.query_document_id,

        "query_clause_id":
            result.query_clause_id,

        "retrieval_rank":
            result.rank,

        "retrieval_score":
            result.score,

        # Preserve original text for debugging/explanation.
        "query_text":
            result.query_text,

        "candidate_text":
            result.candidate_text,

        # Explicitly record whether context was used.
        "query_context_used":
            query_context != result.query_text,

        "candidate_context_used":
            candidate_context != result.candidate_text,
    }
    
# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    print("=" * 70)
    print("LEXCONFLICT END-TO-END PIPELINE")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

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

    selected_document_ids = list(
        documents.keys()
    )[:MAX_DOCUMENTS]

    selected_clauses = []

    for document_id in selected_document_ids:

        selected_clauses.extend(
            documents[document_id]
        )

    print(
        f"Selected documents: "
        f"{len(selected_document_ids)}"
    )

    print(
        f"Selected clauses: "
        f"{len(selected_clauses)}"
    )
    
    # Map clauses by ID for quick lookup throughout the pipeline.
    clause_lookup = {
        str(clause["clause_id"]): clause
        for clause in selected_clauses
    }

    # ========================================================
    # 2. CANDIDATE RETRIEVAL
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 1: CANDIDATE RETRIEVAL")
    print("-" * 70)

    retriever = CandidateRetriever(
        top_k=TOP_K,
        same_document_only=True
    )

    retriever.fit(
        selected_clauses
    )

    retrieval_results = (
        retriever.retrieve_all(
            top_k=TOP_K,
            same_document_only=True
        )
    )

    print(
        f"Retrieval results: "
        f"{len(retrieval_results)}"
    )
    
    print("\n" + "-" * 70)
    print("STEP 2: BUILD NLI PAIRS")
    print("-" * 70)

    pairs = [
        retrieval_to_pair(
            result,
            clause_lookup={
                str(clause["clause_id"]): clause
                for clause in selected_clauses
            }
        )

        for result in retrieval_results
    ]

    pairs = pairs[:100]

    print(
        f"NLI pairs created: "
        f"{len(pairs)}"
    )
    
    context_pairs = sum(
        1
        for pair in pairs
        if pair.get("query_context_used")
        or pair.get("candidate_context_used")
    )

    print(
        f"Pairs using parent context: "
        f"{context_pairs}"
    )

    if not pairs:

        raise RuntimeError(
            "No candidate pairs were generated."
        )
        
    
    for pair in pairs:

        if (
            pair["query_clause_id"] == "3_clause_0026"
            or pair["clause_id"] == "3_clause_0026"
        ):
    
            print("\nContext example:")
            print(
                "Query:",
                pair["hypothesis"]
            )
            print(
                "Evidence:",
                pair["evidence"]
            )
    
            break

    

    # ========================================================
    # 4. LEGAL-BERT NLI
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 3: LEGAL-BERT NLI")
    print("-" * 70)

    nli_model = LegalBERTNLI(
        model_path=MODEL_PATH,
        device="cpu"
    )

    nli_predictions = nli_model.predict(
        pairs,
        batch_size=NLI_BATCH_SIZE,
        max_length=NLI_MAX_LENGTH,
        num_workers=0
    )

    print(
        f"NLI predictions: "
        f"{len(nli_predictions)}"
    )

    if len(nli_predictions) != len(pairs):

        raise RuntimeError(
            "NLI prediction count does not "
            "match pair count."
        )

    # ========================================================
    # 5. STRUCTURED + HYBRID REASONING
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 4: HYBRID CONFLICT REASONING")
    print("-" * 70)

    conflict_engine = ConflictEngine()

    hybrid_reasoner = HybridReasoner(
        nli_weight=NLI_WEIGHT,
        structured_weight=STRUCTURED_WEIGHT,
        conflict_threshold=CONFLICT_THRESHOLD
    )


    hybrid_results = []

    conflict_pairs = []
    
    for index, pair in enumerate(pairs):

        nli_prediction = (
            nli_predictions[index]
        )

        candidate_clause = clause_lookup.get(
            str(pair["clause_id"])
        )

        if candidate_clause is None:
            continue

        query_clause = clause_lookup.get(
            str(pair["query_clause_id"])
        )

        if query_clause is None:
            continue

        structured_result = (
            conflict_engine.analyze_pair(
                query_clause,
                candidate_clause
            )
        )

        nli_result = {
            "label":
                nli_prediction["label"],

            "label_id":
                nli_prediction["label_id"],

            "confidence":
                nli_prediction["confidence"],

            "probabilities":
                nli_prediction["probabilities"]
        }

        hybrid_result = (
            hybrid_reasoner.analyze(
                nli_result=nli_result,
                structured_result=structured_result
            )
        )

        result = {
            "query_clause_id":
                pair["query_clause_id"],

            "candidate_clause_id":
                pair["clause_id"],

            "document_id":
                pair["document_id"],

            "hypothesis":
                pair["hypothesis"],

            "clause_text":
                pair["clause_text"],

            "retrieval_rank":
                pair["retrieval_rank"],

            "retrieval_score":
                pair["retrieval_score"],

            "nli_label":
                nli_prediction["label"],

            "nli_confidence":
                nli_prediction["confidence"],

            "nli_contradiction_probability":
                nli_prediction["probabilities"][2],

            "structured_conflict_score":
                structured_result.confidence,

            "structured_conflict_type":
                structured_result.conflict_type,

            "hybrid_conflict_score":
                hybrid_result["hybrid_conflict_score"],

            "is_conflict":
                hybrid_result["is_conflict"],

            "explanation":
                hybrid_result.get("explanation")
        }

        hybrid_results.append(
            result
        )

        if result["is_conflict"]:

            conflict_pairs.append(
                result
            )

 
    print(
        f"Hybrid results: "
        f"{len(hybrid_results)}"
    )

    print(
        f"Detected conflict pairs: "
        f"{len(conflict_pairs)}"
    )

    # ========================================================
    # 6. BUILD CONFLICT GRAPH
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 5: BUILD CONFLICT GRAPH")
    print("-" * 70)

    graph = ConflictGraph()

    graph_nodes = []

    for clause in selected_clauses:

        node = clause_to_node(
            clause
        )

        graph.add_node(
            node
        )

        graph_nodes.append(
            node
        )

    # --------------------------------------------------------
    # Add conflict edges
    # --------------------------------------------------------

    added_edges = 0

    for result in conflict_pairs:

        source_id = str(
            result["query_clause_id"]
        )

        target_id = str(
            result["candidate_clause_id"]
        )

        if source_id == target_id:
            continue

        edge = GraphEdge(
            source_id=source_id,

            target_id=target_id,

            relationship="conflict",

            confidence=float(
                result[
                    "hybrid_conflict_score"
                ]
            ),

            conflict_type=result[
                "structured_conflict_type"
            ],

            nli_label=result[
                "nli_label"
            ],

            nli_contradiction_probability=float(
                result[
                    "nli_contradiction_probability"
                ]
            ),

            structured_conflict_score=float(
                result[
                    "structured_conflict_score"
                ]
            ),

            hybrid_conflict_score=float(
                result[
                    "hybrid_conflict_score"
                ]
            ),

            explanation=result.get(
                "explanation"
            )
        )

        graph.add_edge(
            edge
        )

        added_edges += 1

    print(
        f"Graph nodes: "
        f"{len(graph.get_nodes())}"
    )

    print(
        f"Graph edges: "
        f"{len(graph.get_edges())}"
    )

    print(
        f"Conflict edges: "
        f"{len(graph.get_conflicts())}"
    )

    # ========================================================
    # 7. CONVERT GRAPH TO DICTIONARIES
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 6: PREPARE GRAPH PROPAGATION")
    print("-" * 70)

    node_dicts = []

    for node in graph.get_nodes():

        node_dicts.append({
            "node_id":
                node.node_id,

            "document_id":
                node.document_id,

            "clause_id":
                node.clause_id,

            "text":
                node.text,

            "party":
                node.party,

            "modality":
                node.modality,

            "condition":
                node.condition,

            "proposition":
                node.proposition,

            "dataset":
                node.dataset,

            "conflict_degree":
                node.conflict_degree
        })

    edge_dicts = []

    for edge in graph.get_edges():

        edge_dicts.append({
            "source_id":
                edge.source_id,

            "target_id":
                edge.target_id,

            "relationship":
                edge.relationship,

            "confidence":
                edge.confidence,

            "conflict_type":
                edge.conflict_type,

            "nli_label":
                edge.nli_label,

            "nli_contradiction_probability":
                edge.nli_contradiction_probability,

            "structured_conflict_score":
                edge.structured_conflict_score,

            "hybrid_conflict_score":
                edge.hybrid_conflict_score,

            "explanation":
                edge.explanation
        })

    # ========================================================
    # 8. GRAPH PROPAGATION
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 7: CONFLICT PROPAGATION")
    print("-" * 70)

    propagation = GraphPropagation(
        alpha=PROPAGATION_ALPHA,
        iterations=PROPAGATION_ITERATIONS
    )

    propagation_results = (
        propagation.propagate(
            node_dicts,
            edge_dicts
        )
    )

    print(
        f"Propagation results: "
        f"{len(propagation_results)}"
    )

    # ========================================================
    # 9. RISK SCORING
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 8: CLAUSE-LEVEL RISK")
    print("-" * 70)

    risk_engine = RiskEngine(
        propagated_weight=PROPAGATED_WEIGHT,
        direct_weight=DIRECT_WEIGHT,
        connectivity_weight=CONNECTIVITY_WEIGHT
    )

    propagation_data = {
        "nodes": list(
            propagation_results.values()
        )
    }

    graph_data = {
        "nodes": node_dicts,
        "edges": edge_dicts
    }

    risk_results = (
        risk_engine.calculate_graph_risk(
            propagation_data,
            graph_data
        )
    )

    print(
        f"Risk results: "
        f"{len(risk_results)}"
    )

    # ========================================================
    # 10. SORT TOP RISKS
    # ========================================================

    risk_results_sorted = sorted(
        risk_results,
        key=lambda result:
            result.risk_score,
        reverse=True
    )

    print("\nTOP RISK CLAUSES")
    print("-" * 70)

    for index, result in enumerate(
        risk_results_sorted[:10],
        start=1
    ):

        print(
            f"{index}. "
            f"{result.node_id} | "
            f"risk={result.risk_score:.4f} | "
            f"level={result.risk_level}"
        )

    # ========================================================
    # 11. SAVE PIPELINE RESULTS
    # ========================================================

    print("\n" + "-" * 70)
    print("STEP 9: SAVE RESULTS")
    print("-" * 70)

    output = {
        "configuration": {
            "model_path":
                MODEL_PATH,

            "max_documents":
                MAX_DOCUMENTS,

            "top_k":
                TOP_K,

            "nli_weight":
                NLI_WEIGHT,

            "structured_weight":
                STRUCTURED_WEIGHT,

            "conflict_threshold":
                CONFLICT_THRESHOLD,

            "propagation_alpha":
                PROPAGATION_ALPHA,

            "propagation_iterations":
                PROPAGATION_ITERATIONS,

            "propagated_weight":
                PROPAGATED_WEIGHT,

            "direct_weight":
                DIRECT_WEIGHT,

            "connectivity_weight":
                CONNECTIVITY_WEIGHT
        },

        "dataset": {
            "documents_available":
                len(documents),

            "documents_processed":
                len(selected_document_ids),

            "clauses_processed":
                len(selected_clauses),

            "candidate_pairs":
                len(pairs),

            "conflict_pairs":
                len(conflict_pairs)
        },

        "graph": {
            "number_of_nodes":
                len(graph.get_nodes()),

            "number_of_edges":
                len(graph.get_edges()),

            "number_of_conflict_edges":
                len(graph.get_conflicts())
        },

        "hybrid_results":
            hybrid_results,

        "propagation_results":
            propagation_results,

        "risk_results": [
            result.to_dict()
            for result in risk_results_sorted
        ]
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "Pipeline output saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "✓ LEXCONFLICT END-TO-END PIPELINE COMPLETE"
    )

    return output


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_pipeline()