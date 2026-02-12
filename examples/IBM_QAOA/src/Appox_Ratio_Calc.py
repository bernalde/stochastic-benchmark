from pathlib import Path
import json

def get_minmax(minmax_path: str, graph_type: str, instance: str, num_nodes: str, 
                ER_probability: str = None, swap_layers: str = None, degree: str = None):
    """Locate the min/max-cut JSON file for a given instance.

    Parameters
    ----------
    minmax_path : str
        Base directory containing per-graph-type subfolders.
    graph_type : str
        Graph family identifier.
    instance : str
        Instance identifier used in the filename pattern.
    num_nodes : str
        Number of nodes encoded in the filename.
    ER_probability : str, optional
        Erdos-Renyi probability suffix used for `graph_type="erdos_renyi"`.
    swap_layers : str, optional
        Swap-layer count used for `graph_type="line_to_full"`.
    degree : str, optional
        Degree used for `graph_type="random_regular"`.

    Returns
    -------
    pathlib.Path
        Path to the unique matching min/max-cut JSON file.

    Raises
    ------
    ValueError
        If `graph_type` is not supported.
    FileNotFoundError
        If no matching file is found.
    RuntimeError
        If multiple matching files are found.
    """

    if graph_type == "heavy_hex":
        instance_path = f"{graph_type}/{instance}*heavyhex_{num_nodes}nodes*.json"
    elif graph_type == "erdos_renyi":
        instance_path = f"{graph_type}/{instance}_{num_nodes}nodes_erdosrenyi{ER_probability}percent*.json"
    elif graph_type == "line_to_full":
        instance_path = f"{graph_type}/{instance}_{num_nodes}nodes_{swap_layers}swap_layers*.json"
    elif graph_type == "random_regular":
        instance_path = f"{graph_type}/{instance}_{num_nodes}nodes_random{degree}regular*.json"
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    minmax_instance_paths = list(Path(minmax_path).glob(instance_path))

    if len(minmax_instance_paths) == 0:
        raise FileNotFoundError(f"No minmax file for instance {instance}")
    if len(minmax_instance_paths) > 1:
        raise RuntimeError(f"Multiple minmax files for instance {instance}")

    return minmax_instance_paths[0]

def extract_minmax_args(minmax_instance_paths: Path):

    """Extract min-cut, max-cut, and sum-of-weights values from a JSON file.

    Parameters
    ----------
    minmax_instance_paths : pathlib.Path
        Path to a min/max-cut JSON file.

    Returns
    -------
    tuple of (float, float, float) or None
        `(min_cut, max_cut, sum_of_weights)` if all keys exist; otherwise None.
    """

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
    """Compute the MaxCut approximation ratio for a given energy.

    Parameters
    ----------
    min_cut : float
        Minimum cut value for the instance.
    max_cut : float
        Maximum cut value for the instance.
    sum_weights : float
        Sum of edge weights for the instance.
    energy : float
        Energy returned by the MaximumCut objective.

    Returns
    -------
    float
        Approximation ratio computed from the implied cut value.
    """
    
    cut_val = energy + 0.5 * sum_weights

    return (cut_val - min_cut) / (max_cut - min_cut)