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

def explain_statistics(result):
    mean = result["mean"]
    median = result["median"]

    if isinstance(result["std_dev"], str):
        spread_hint = "There wasn't enough data to measure how spread out the numbers are."
    else:
        spread_hint = (
            f"The standard deviation ({result['std_dev']}) tells you how far, on average, "
            f"each number strays from the mean — a small number means the data is tightly "
            f"clustered, a large one means it's spread out."
        )

    if abs(mean - median) < 0.01:
        shape_hint = "Since the mean and median are nearly equal, your data looks fairly symmetric (no strong skew)."
    elif mean > median:
        shape_hint = "Since the mean is higher than the median, a few unusually large values are pulling the average up."
    else:
        shape_hint = "Since the mean is lower than the median, a few unusually small values are pulling the average down."

    return (
        f"The mean ({mean}) is the regular average — add everything up and divide by "
        f"how many numbers there are. The median ({median}) is the true middle value "
        f"when the numbers are sorted — less thrown off by extreme outliers than the mean. "
        f"The mode ({result['mode']}) is simply whichever value appears most often. "
        f"{spread_hint} {shape_hint}"
    )