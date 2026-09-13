from pathlib import Path
from typing import Any, Dict, List
import re
from app.utils.uploaded_clause_parser import (
    parse_uploaded_contract
)

from app.utils.temporal_conflict import (
    analyze_temporal_conflict
)

from src.retrieval.candidate_retrieval import (
    CandidateRetriever
)

from src.models.nli_inference import (
    LegalBERTNLI
)

from src.reasoning.conflict_engine import (
    ConflictEngine
)

from src.reasoning.hybrid_reasoner import (
    HybridReasoner
)

from src.graph.graph_node import (
    GraphNode
)

from src.graph.graph_edge import (
    GraphEdge
)

from src.graph.conflict_graph import (
    ConflictGraph
)

from src.graph.graph_propagation import (
    GraphPropagation
)

from src.risk.risk_engine import (
    RiskEngine
)

from src.resolution.resolution_engine import (
    ConflictResolutionEngine
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lexconflict_legalbert"
)


# ============================================================
# MODEL / REASONING CONFIGURATION
# ============================================================

# ============================================================
# UPLOADED CONTRACT HYBRID CONFIGURATION
# ============================================================

NLI_WEIGHT = 0.25

STRUCTURED_WEIGHT = 0.25

TEMPORAL_WEIGHT = 0.50

CONFLICT_THRESHOLD = 0.50


PROPAGATION_ALPHA = 0.70

PROPAGATION_ITERATIONS = 3


PROPAGATED_WEIGHT = 0.50

DIRECT_WEIGHT = 0.30

CONNECTIVITY_WEIGHT = 0.20


# ============================================================
# HELPER
# ============================================================

def clause_to_node(
    clause: Dict[str, Any]
) -> GraphNode:

    return GraphNode(

        node_id=str(
            clause["clause_id"]
        ),

        document_id=str(
            clause["document_id"]
        ),

        clause_id=str(
            clause["clause_id"]
        ),

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
# REMOVE DUPLICATE CONFLICT PAIRS
# ============================================================

def deduplicate_conflicts(
    conflicts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    best = {}

    for item in conflicts:

        clause_a = str(
            item["query_clause_id"]
        )

        clause_b = str(
            item["candidate_clause_id"]
        )

        key = tuple(
            sorted(
                [clause_a, clause_b]
            )
        )

        score = float(
            item["hybrid_conflict_score"]
        )

        if (
            key not in best
            or score >
            best[key]["hybrid_conflict_score"]
        ):

            best[key] = item

    return list(
        best.values()
    )


# ============================================================
# MAIN CONTRACT ANALYSIS
# ============================================================

def analyze_contract(
    text: str,
    document_name: str,
    nli_model: LegalBERTNLI,
    top_k: int = 3,
    max_clauses: int = 120,
    max_pairs: int = 300
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not text or not text.strip():

        raise ValueError(
            "The uploaded contract does not "
            "contain readable text."
        )


    # --------------------------------------------------------
    # Create document ID
    # --------------------------------------------------------

    document_id = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        Path(document_name).stem
    ).strip("_")

    if not document_id:

        document_id = "uploaded_contract"


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    # --------------------------------------------------------
    # UPLOADED CONTRACT CLAUSE PARSING
    # --------------------------------------------------------

    clause_objects = parse_uploaded_contract(

        text=text,

        document_id=document_id,

        dataset="uploaded_contract"
    )


    clauses = [

        clause.to_dict()

        for clause in clause_objects
    ]


    if not clauses:

        raise ValueError(
            "No clauses could be extracted "
            "from this contract."
        )


    # --------------------------------------------------------
    # Limit clauses for initial application
    # --------------------------------------------------------

    clauses = clauses[
        :max_clauses
    ]


    clause_lookup = {

        str(clause["clause_id"]):
            clause

        for clause in clauses
    }


    # ========================================================
    # CANDIDATE RETRIEVAL
    # ========================================================

    retriever = CandidateRetriever(

        top_k=top_k,

        same_document_only=True
    )


    retriever.fit(
        clauses
    )


    retrieval_results = (
        retriever.retrieve_all(

            top_k=top_k,

            same_document_only=True
        )
    )


    # ========================================================
    # BUILD NLI PAIRS
    # ========================================================

    pairs = []


    for result in retrieval_results:

        if len(pairs) >= max_pairs:

            break


        query_clause = clause_lookup[
            str(
                result.query_clause_id
            )
        ]


        candidate_clause = clause_lookup[
            str(
                result.candidate_clause_id
            )
        ]


        query_context = (
            query_clause.get(
                "context_text"
            )
            or
            query_clause.get(
                "text",
                ""
            )
        )


        candidate_context = (
            candidate_clause.get(
                "context_text"
            )
            or
            candidate_clause.get(
                "text",
                ""
            )
        )


        pairs.append({

            "hypothesis":
                query_context,

            "evidence":
                candidate_context,

            "clause_text":
                candidate_clause.get(
                    "text",
                    ""
                ),

            "clause_id":
                str(
                    result.candidate_clause_id
                ),

            "document_id":
                document_id,

            "query_clause_id":
                str(
                    result.query_clause_id
                ),

            "retrieval_rank":
                result.rank,

            "retrieval_score":
                float(
                    result.score
                ),

            "query_text":
                query_clause.get(
                    "text",
                    ""
                ),

            "candidate_text":
                candidate_clause.get(
                    "text",
                    ""
                )
        })


    if not pairs:

        raise ValueError(
            "No candidate clause pairs "
            "were generated."
        )


    # ========================================================
    # VERIFY MODEL LABEL MAPPING
    # ========================================================

    label_2 = str(
        nli_model.id2label.get(
            2,
            ""
        )
    ).lower()


    if "contrad" not in label_2:

        raise RuntimeError(

            "The Legal-BERT model label mapping "
            "is incompatible with the current "
            "hybrid reasoner.\n\n"

            f"Detected mapping: "
            f"{nli_model.id2label}\n\n"

            "The current project expects "
            "label ID 2 to represent "
            "Contradiction."
        )


    # ========================================================
    # LEGAL-BERT NLI
    # ========================================================

    nli_predictions = nli_model.predict(

        pairs,

        batch_size=16,

        max_length=120,

        num_workers=0
    )


    # ========================================================
    # HYBRID REASONING
    # ========================================================

    conflict_engine = (
        ConflictEngine()
    )


    hybrid_reasoner = (
        HybridReasoner(

            nli_weight=NLI_WEIGHT,

            structured_weight=
                STRUCTURED_WEIGHT,

            temporal_weight=
                TEMPORAL_WEIGHT,

            conflict_threshold=
                CONFLICT_THRESHOLD
        )
    )


    all_results = []

    conflicts = []


    for pair, prediction in zip(
        pairs,
        nli_predictions
    ):

        query_clause = clause_lookup[
            pair["query_clause_id"]
        ]


        candidate_clause = clause_lookup[
            pair["clause_id"]
        ]

        # --------------------------------------------------------
        # STRUCTURED REASONING
        # --------------------------------------------------------
        
        structured_result = (
            conflict_engine.analyze_pair(
            
                query_clause,
        
                candidate_clause
            )
        )
        
        
        # --------------------------------------------------------
        # TEMPORAL / NUMERIC REASONING
        # --------------------------------------------------------
        
        temporal_result = (
            analyze_temporal_conflict(
            
                query_clause,
        
                candidate_clause
            )
        )
        
        
        # --------------------------------------------------------
        # HYBRID REASONING
        # --------------------------------------------------------
        
        hybrid_result = (
            hybrid_reasoner.analyze(
            
                nli_result=prediction,
        
                structured_result=
                    structured_result,
        
                temporal_result=
                    temporal_result
            )
        )


        result = {

            "query_clause_id":
                pair["query_clause_id"],

            "candidate_clause_id":
                pair["clause_id"],

            "document_id":
                document_id,

            "query_text":
                pair["query_text"],

            "candidate_text":
                pair["candidate_text"],

            "hypothesis":
                pair["hypothesis"],

            "evidence":
                pair["evidence"],

            "retrieval_rank":
                pair["retrieval_rank"],

            "retrieval_score":
                pair["retrieval_score"],

            "nli_label":
                prediction["label"],

            "nli_confidence":
                float(
                    prediction["confidence"]
                ),

            "nli_contradiction_probability":
                float(
                    prediction["probabilities"][2]
                ),

            "structured_conflict_score":
                float(
                    structured_result.confidence
                ),

            "structured_conflict_type":
                structured_result.conflict_type,

            "temporal_conflict_score":
                float(
                    temporal_result[
                        "temporal_conflict_score"
                    ]
                ),

            "temporal_conflict_type":
                temporal_result[
                    "conflict_type"
                ],

            "temporal_explanation":
                temporal_result[
                    "explanation"
                ],


            "hybrid_conflict_score":
                float(
                    hybrid_result[
                        "hybrid_conflict_score"
                    ]
                ),

            "is_conflict":
                bool(
                    hybrid_result[
                        "is_conflict"
                    ]
                ),

            "confidence_level":
                hybrid_result.get(
                    "confidence_level",
                    "Low"
                ),

            "conflict_type":
                hybrid_result.get(
                    "conflict_type",
                    structured_result.conflict_type
                ),


            "explanation":
                hybrid_result.get(
                    "explanation",
                    ""
                )
        }


        all_results.append(
            result
        )


        if (
            result["is_conflict"]
            and result["conflict_type"] != "no_clear_conflict"
        ):

            conflicts.append(
                result
            )


    # Remove A→B / B→A duplicates
    conflicts = deduplicate_conflicts(
        conflicts
    )


    # ========================================================
    # BUILD CONFLICT GRAPH
    # ========================================================

    graph = ConflictGraph()


    for clause in clauses:

        graph.add_node(
            clause_to_node(
                clause
            )
        )


    edge_dicts = []

    resolution_engine = (
        ConflictResolutionEngine()
    )

    resolutions = []


    for result in conflicts:

        edge = GraphEdge(

            source_id=
                result[
                    "query_clause_id"
                ],

            target_id=
                result[
                    "candidate_clause_id"
                ],

            relationship="conflict",

            confidence=
                result[
                    "hybrid_conflict_score"
                ],

            conflict_type=
                result[
                    "conflict_type"
                ],

            nli_label=
                result[
                    "nli_label"
                ],

            nli_contradiction_probability=
                result[
                    "nli_contradiction_probability"
                ],

            structured_conflict_score=
                result[
                    "structured_conflict_score"
                ],

            hybrid_conflict_score=
                result[
                    "hybrid_conflict_score"
                ],

            explanation=
                result.get(
                    "explanation"
                )
        )


        graph.add_edge(
            edge
        )


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


        resolution = (
            resolution_engine.resolve(
                edge
            )
        )


        resolutions.append({

            "source_id":
                edge.source_id,

            "target_id":
                edge.target_id,

            "severity":
                resolution.severity,

            "recommendation":
                resolution.recommendation,

            "explanation":
                resolution.explanation,

            "suggested_revision":
                resolution.suggested_revision
        })


    # ========================================================
    # GRAPH NODE DATA
    # ========================================================

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


    # ========================================================
    # GRAPH PROPAGATION
    # ========================================================

    propagation = (
        GraphPropagation(

            alpha=
                PROPAGATION_ALPHA,

            iterations=
                PROPAGATION_ITERATIONS
        )
    )


    propagation_results = (
        propagation.propagate(

            node_dicts,

            edge_dicts
        )
    )


    # ========================================================
    # RISK ENGINE
    # ========================================================

    risk_engine = RiskEngine(

        propagated_weight=
            PROPAGATED_WEIGHT,

        direct_weight=
            DIRECT_WEIGHT,

        connectivity_weight=
            CONNECTIVITY_WEIGHT
    )


    risk_results = (
        risk_engine.calculate_graph_risk(

            {
                "nodes":
                    list(
                        propagation_results.values()
                    )
            },

            {
                "nodes":
                    node_dicts,

                "edges":
                    edge_dicts
            }
        )
    )


    risk_results = sorted(

        risk_results,

        key=lambda result:
            result.risk_score,

        reverse=True
    )


    risk_summary = (
        risk_engine.summarize_risk(
            risk_results
        )
    )


    # ========================================================
    # COMBINE CLAUSES + RISK
    # ========================================================

    risk_by_id = {

        result.node_id:
            result.to_dict()

        for result in risk_results
    }


    clause_rows = []


    for clause in clauses:

        risk = risk_by_id.get(

            clause["clause_id"],

            {}
        )


        clause_rows.append({

            **clause,

            "risk_score":
                risk.get(
                    "risk_score",
                    0.0
                ),

            "risk_level":
                risk.get(
                    "risk_level",
                    "minimal"
                ),

            "conflict_degree":
                risk.get(
                    "conflict_degree",
                    0
                )
        })


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "document_name":
            document_name,

        "document_id":
            document_id,

        "clauses":
            clause_rows,

        "retrieval_results":
            all_results,

        "conflicts":
            conflicts,

        "resolutions":
            resolutions,

        "propagation_results":
            propagation_results,

        "risk_results": [

            result.to_dict()

            for result in risk_results
        ],

        "risk_summary":
            risk_summary,

        "graph": {

            "nodes":
                node_dicts,

            "edges":
                edge_dicts
        },

        "configuration": {

            "top_k":
                top_k,

            "max_clauses":
                max_clauses,

            "max_pairs":
                max_pairs,

            "nli_weight":
                NLI_WEIGHT,

            "structured_weight":
                STRUCTURED_WEIGHT,

            "temporal_weight":
                TEMPORAL_WEIGHT,

            "conflict_threshold":
                CONFLICT_THRESHOLD
        }
    }