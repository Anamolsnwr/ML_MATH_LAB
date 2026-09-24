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