from src.graph.graph_edge import GraphEdge
from src.resolution.resolution_engine import (
    ConflictResolutionEngine
)


def main():

    edge = GraphEdge(

        source_id="A",

        target_id="B",

        relationship="conflict",

        confidence=0.90,

        conflict_type="obligation_vs_prohibition",

        nli_label="Contradiction",

        nli_contradiction_probability=0.95,

        structured_conflict_score=0.80,

        hybrid_conflict_score=0.88,

        explanation="Detected conflict."
    )

    engine = ConflictResolutionEngine()

    result = engine.resolve(edge)

    print(result)

    assert result.severity == "High"

    print("\n✓ RESOLUTION ENGINE TEST PASSED")


if __name__ == "__main__":
    main()