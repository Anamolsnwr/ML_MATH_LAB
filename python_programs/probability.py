def calculate_probability(favorable, total):
    try:
        fav = float(favorable)
        tot = float(total)
    except (ValueError, TypeError):
        raise ValueError("Inputs must be numeric values.")

    if tot <= 0:
        raise ValueError("Total outcomes must be greater than 0.")
    if fav < 0:
        raise ValueError("Favorable outcomes cannot be negative.")
    if fav > tot:
        raise ValueError("Favorable outcomes cannot exceed total outcomes.")

    prob = fav / tot
    return {
        "favorable": int(fav) if fav.is_integer() else fav,
        "total": int(tot) if tot.is_integer() else tot,
        "probability": round(prob, 4),
        "percentage": f"{round(prob * 100, 2)}%"
    }

def explain_probability(result):
    pct = result["percentage"]
    if pct >= 75:
        odds_feel = "very likely to happen"
    elif pct >= 50:
        odds_feel = "more likely than not to happen"
    elif pct >= 25:
        odds_feel = "possible, but more likely not to happen"
    else:
        odds_feel = "unlikely to happen"

    return (
        f"Probability is simply: (ways the thing you want can happen) divided by "
        f"(all the things that could possibly happen). Here, out of {result['total']} "
        f"total possible outcomes, {result['favorable']} count as a 'win'. "
        f"That works out to {result['probability']} (or {pct}%), meaning this event is {odds_feel}."
    )