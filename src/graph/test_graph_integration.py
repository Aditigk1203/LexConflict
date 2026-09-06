from src.models.nli_inference import LegalBERTNLI
from src.reasoning.conflict_engine import ConflictEngine
from src.reasoning.hybrid_reasoner import HybridReasoner

from src.graph.graph_node import GraphNode
from src.graph.conflict_graph import ConflictGraph


MODEL_PATH = "models/lexconflict_legalbert"


def main():

    print("=" * 70)
    print("LEXCONFLICT PHASE 5 → PHASE 6 INTEGRATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load NLI model
    # ---------------------------------------------------------

    nli_model = LegalBERTNLI(
        model_path=MODEL_PATH,
        device="cpu"
    )

    # ---------------------------------------------------------
    # 2. Create clauses
    # ---------------------------------------------------------

    hypothesis = (
        "The Supplier shall provide "
        "the report within 30 days."
    )

    clause = (
        "The Supplier shall not provide "
        "the report within 30 days."
    )

    # ---------------------------------------------------------
    # 3. NLI
    # ---------------------------------------------------------

    nli_prediction = nli_model.predict(
        [
            {
                "hypothesis": hypothesis,
                "clause_text": clause
            }
        ],
        batch_size=1,
        max_length=120,
        num_workers=0
    )[0]

    print("\nNLI:")
    print(nli_prediction)

    # ---------------------------------------------------------
    # 4. Structured reasoning
    # ---------------------------------------------------------

    clause_a = {
        "clause_id": "doc1_clause_001",
        "text": hypothesis,
        "party": "supplier",
        "modality": "obligation",
        "condition": None
    }

    clause_b = {
        "clause_id": "doc1_clause_002",
        "text": clause,
        "party": "supplier",
        "modality": "prohibition",
        "condition": None
    }

    conflict_engine = ConflictEngine()

    structured_result = (
        conflict_engine.analyze_pair(
            clause_a,
            clause_b
        )
    )

    print("\nStructured:")
    print(structured_result)

    # ---------------------------------------------------------
    # 5. Hybrid reasoning
    # ---------------------------------------------------------

    hybrid_reasoner = HybridReasoner(
        nli_weight=0.95,
        structured_weight=0.05,
        conflict_threshold=0.71
    )

    hybrid_result = hybrid_reasoner.analyze(
        nli_result=nli_prediction,
        structured_result=structured_result
    )

    print("\nHybrid:")
    print(hybrid_result)

    # ---------------------------------------------------------
    # 6. Create graph
    # ---------------------------------------------------------

    graph = ConflictGraph()

    node_a = GraphNode(
        node_id="doc1_clause_001",
        document_id="doc1",
        clause_id="doc1_clause_001",
        text=hypothesis,
        party="supplier",
        modality="obligation"
    )

    node_b = GraphNode(
        node_id="doc1_clause_002",
        document_id="doc1",
        clause_id="doc1_clause_002",
        text=clause,
        party="supplier",
        modality="prohibition"
    )

    graph.add_node(node_a)
    graph.add_node(node_b)

    # ---------------------------------------------------------
    # 7. Add hybrid result to graph
    # ---------------------------------------------------------

    edge = graph.add_hybrid_result(
        source_id="doc1_clause_001",
        target_id="doc1_clause_002",
        hybrid_result=hybrid_result
    )

    print("\nGRAPH EDGE:")
    print(edge)

    # ---------------------------------------------------------
    # 8. Validation
    # ---------------------------------------------------------

    assert graph.number_of_nodes() == 2

    assert graph.number_of_edges() == 1

    assert edge.source_id == "doc1_clause_001"

    assert edge.target_id == "doc1_clause_002"

    assert edge.relationship in [
        "conflict",
        "related"
    ]

    assert (
        edge.hybrid_conflict_score
        == hybrid_result[
            "hybrid_conflict_score"
        ]
    )

    print("\nGRAPH SUMMARY:")

    print(
        graph.summary()
    )

    print(
        "\n✓ PHASE 5 → PHASE 6 "
        "INTEGRATION TEST PASSED"
    )


if __name__ == "__main__":
    main()