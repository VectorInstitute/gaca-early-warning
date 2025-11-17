"""Configuration management and argument parsing utilities.

This module provides functions for loading and parsing configuration files and
command-line arguments for the GACA early warning pipeline. Configuration data is
loaded from YAML files and converted to namespace objects for easy attribute access.

Functions
---------
get_args
    Parse command-line arguments and load configuration from YAML file.

Notes
-----
The configuration is expected to be provided as a YAML file, which is recursively
converted to nested Namespace objects for convenient dotted-notation access.
"""
###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# purpose: to set up config + args for pipeline
###########################################################################################

from argparse import ArgumentParser, Namespace
from typing import Any

import yaml


def get_args() -> Namespace:
    """Parse command-line arguments and load configuration from YAML file.

    This function parses the command-line argument for the configuration file path,
    loads the YAML configuration, and converts it to a nested Namespace object for
    convenient dotted-notation access to configuration parameters.

    Returns
    -------
    argparse.Namespace
        Configuration loaded from YAML as a nested Namespace object. Each nested
        dictionary in the YAML is recursively converted to a Namespace, allowing
        access to configuration values via dot notation (e.g., cfg.model.hidden_dim).

    Notes
    -----
    Requires a command-line argument '--config' pointing to a valid YAML
    configuration file.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    def to_namespace(obj: Any) -> Any:
        if isinstance(obj, dict):
            ns = Namespace()
            for key, value in obj.items():
                setattr(ns, key, to_namespace(value))
            return ns
        return obj

    return to_namespace(cfg)
