import statistics as st

def calculate_statistics(numbers_str):
    clean_str = str(numbers_str).replace(',', ' ').split()
    if not clean_str:
        raise ValueError("Dataset cannot be empty. Please enter numbers.")

    try:
        data = [float(x) for x in clean_str]
    except ValueError:
        raise ValueError("Dataset contains invalid numbers. Use space or comma separated values.")

    mean_val = round(st.mean(data), 4)
    median_val = round(st.median(data), 4)

    try:
        mode_val = round(st.mode(data), 4)
    except st.StatisticsError:
        mode_val = "No unique mode"

    variance_val = round(st.variance(data), 4) if len(data) > 1 else 0.0
    stdev_val = round(st.stdev(data), 4) if len(data) > 1 else 0.0

    return {
        "data": data,
        "count": len(data),
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "variance": variance_val,
        "stdev": stdev_val
    }