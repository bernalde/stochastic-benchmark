import re
import pandas as pd
import numpy as np


def is_empty_nested_list(x):
    """Check whether `x` is a non-empty list of empty lists.

    Parameters
    ----------
    x : object
        Value to check.

    Returns
    -------
    bool
        True if `x` is a list with at least one element and every element is an empty list.
    """
    return isinstance(x, list) and len(x) > 0 and all(isinstance(i, list) and len(i) == 0 for i in x)

def parse_instance_name(name: str) -> pd.Series:
    """ Extracts relevant details about instance such as instance_number, 
        N, graph_type, graph_param, trainer_label, evaluator_label, p

    Parameters
    ----------
    name: string
         instance file name for extraction

    Returns
    -------
    Pandas.Series(Dict(str))
        {"instance_number": instance_number,
        "N": N,
        "graph_type": graph_type,
        "graph_param": graph_param,
        "trainer_label": trainer_label,
        "evaluator_label": evaluator_label,
        "p": p
        }
    """	
    name = str(name).replace(".json", "")

    graph_part, config_part = name.split("_MC_", 1)

    # GRAPH PART: 000N144HH73
    m = re.match(r"(?P<instance_number>\d{3})N(?P<N>\d+)(?P<graph_code>[A-Z]+)(?P<graph_param>\d+)", graph_part)
    if not m:
        return pd.Series({
            "instance_number": np.nan, "N": np.nan, "graph_type": np.nan, "graph_param": np.nan,
            "trainer_label": np.nan, "evaluator_label": np.nan, "p": np.nan
        })

    instance_number = m.group("instance_number")
    N = int(m.group("N"))
    graph_code = m.group("graph_code")
    graph_param = int(m.group("graph_param"))

    graph_map = {"HH": "heavy_hex", "ER": "erdos_renyi", "L2S": "line_to_full", "R": "random_regular"}
    graph_type = graph_map.get(graph_code, graph_code)

    # CONFIG PART: TQA_PP_optMW6_5  OR  LR_MPSAer_opt_DB24_10, etc.
    tokens = config_part.split("_")
    trainer_label = tokens[0] if len(tokens) >= 1 else np.nan
    evaluator_label = tokens[1] if len(tokens) >= 2 else np.nan
    p = int(tokens[-1]) if tokens and tokens[-1].isdigit() else np.nan

    return pd.Series({
        "instance_number": instance_number,
        "N": N,
        "graph_type": graph_type,
        "graph_param": graph_param,
        "trainer_label": trainer_label,
        "evaluator_label": evaluator_label,
        "p": p,
    })