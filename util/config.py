###########################################################################################
# authored july 6, 2025 (to make modular) by jelshawa
# purpose: to set up config + args for pipeline
###########################################################################################

import yaml
from argparse import ArgumentParser, Namespace

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    def to_namespace(obj):
        if isinstance(obj, dict):
            ns = Namespace()
            for key, value in obj.items():
                setattr(ns, key, to_namespace(value))
            return ns
        else:
            return obj

    return to_namespace(cfg)