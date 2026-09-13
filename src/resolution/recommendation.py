def generate_recommendation(
    conflict_type
):

    recommendations = {

        "obligation_vs_prohibition":
            (
                "Review the clauses and determine "
                "whether the action is mandatory "
                "or prohibited."
            ),

        "prohibition_vs_obligation":
            (
                "Clarify whether the action is "
                "permitted or forbidden."
            ),

        "permission_vs_prohibition":
            (
                "Clarify the circumstances under "
                "which the action is permitted "
                "or prohibited."
            ),

        "prohibition_vs_permission":
            (
                "Clarify the circumstances under "
                "which the action is permitted "
                "or prohibited."
            ),

        "positive_vs_negative":
            (
                "Remove contradictory affirmative "
                "and negative statements."
            ),

        "temporal_deadline_conflict":
            (
                "Standardize the conflicting time "
                "periods so that the contract contains "
                "one clearly defined deadline."
            ),

        "potential_semantic_conflict":
            (
                "Review the related clauses manually "
                "and clarify any inconsistent wording."
            ),

        "no_clear_conflict":
            (
                "Manual legal review recommended."
            ),
    }


    return recommendations.get(

        conflict_type,

        "Manual legal review required."
    )