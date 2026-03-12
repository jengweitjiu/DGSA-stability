"""
Configuration loading and shared utilities.
"""

import os

import numpy as np
import yaml


def load_config(path=None):
    """
    Load simulation parameters from config/params.yaml.

    Parameters
    ----------
    path : str or None
        Path to params.yaml. If None, uses config/params.yaml
        relative to the repository root.

    Returns
    -------
    config : dict
        Parsed YAML configuration.
    """
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "config", "params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def numpy_converter(obj):
    """
    JSON-serialization hook for numpy types.

    Usage: ``json.dump(data, f, default=numpy_converter)``
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, np.bool_)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
