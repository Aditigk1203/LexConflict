def generate_recommendation(conflict_type):

    recommendations = {

        "obligation_vs_prohibition":
            "Review clauses and determine whether the action is mandatory or prohibited.",

        "positive_vs_negative":
            "Remove contradictory affirmative and negative statements.",

        "prohibition_vs_obligation":
            "Clarify whether the action is permitted or forbidden.",

        "no_clear_conflict":
            "Manual legal review recommended."
    }

    return recommendations.get(
        conflict_type,
        "Manual legal review required."
    )