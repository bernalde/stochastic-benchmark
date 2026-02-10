def get_minmax(minmax_path: str = None, graph_type: str = None, instance: str = None, num_nodes: str = None, 
                ER_probability: str = None, swap_layers: str = None, degree: str = None):

    if graph_type == "heavy_hex":
        instance_path = f"00{instance}*heavyhex_{num_nodes}nodes*.json"
    elif graph_type == "erdos_renyi":
        instance_path = f"00{instance}_{num_nodes}nodes_erdosrenyi{ER_probability}percent*.json"
    elif graph_type == "line_to_full":
        instance_path = f"00{instance}_{num_nodes}nodes_{swap_layers}swap_layers*.json"
    elif graph_type == "random_regular":
        instance_path = f"00{instance}_{num_nodes}nodes_random{degree}regular*.json"
    else:
        return []

    minmax_instance_paths = list(Path(minmax_path).glob(instance_path))

    return minmax_instance_paths

def extract_minmax_args(minmax_instance_paths: str):

    with minmax_instance_paths.open("r") as f:
        content = json.load(f)

    if "min_cut" not in content or "max_cut" not in content or "sum_of_weights" not in content:
        return None
    min_cut = content["min_cut"]
    max_cut = content["max_cut"]
    sum_weights = content["sum_of_weights"]

    return min_cut, max_cut, sum_weights

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