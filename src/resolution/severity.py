def calculate_severity(conflict_score):

    if conflict_score >= 0.90:
        return "Critical"

    if conflict_score >= 0.70:
        return "High"

    if conflict_score >= 0.50:
        return "Medium"

    return "Low"