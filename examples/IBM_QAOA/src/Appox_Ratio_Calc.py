def maxcut_approximation_ratio(
    min_cut: float, max_cut: float, sum_weights: float, energy: float
) -> float:
    """Compute the approximation ratio for a given max- and min-cuts, sum of
       edge weights, and energy.

    Args:
        min_cut: The minimum cut of a graph.
        max_cut: The maximum cut of a graph.
        sum_weights: The sum of edge-weights in a graph.
        energy: The energy from the MaximumCut problem.

    Returns:
        The approximation ratio for the given energy and associated graph.
    """
    cut_val = energy + 0.5 * sum_weights

    return (cut_val - min_cut) / (max_cut - min_cut)