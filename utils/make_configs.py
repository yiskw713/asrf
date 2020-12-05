import argparse
import dataclasses
import itertools
import os
import sys

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from typing import Any, Dict, List, Optional, Tuple

from libs.config import Config


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(description="make configuration yaml files.")

    parser.add_argument(
        "--root_dir",
        type=str,
        default="./result",
        help="path to a directory where you want to make config files and directories.",
    )

    fields = dataclasses.fields(Config)

    for field in fields:
        if isinstance(field.default, dataclasses._MISSING_TYPE):
            # default value is not set.
            parser.add_argument(
                f"--{field.name}",
                type=field.type,
                nargs="*",
                required=True,
            )
        else:
            # default value is provided in config dataclass.
            parser.add_argument(
                f"--{field.name}",
                type=field.type,
                nargs="*",
                default=field.default,
            )

    return parser.parse_args()


def parse_params(
    args_dict: Dict[str, Any], tuple_object_keys: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], List[str], List[List[Any]]]:

    base_config = {}
    variable_keys = []
    variable_values = []

    for k, v in args_dict.items():
        if isinstance(v, list):
            variable_keys.append(k)
            variable_values.append(v)
        else:
            base_config[k] = v

    return base_config, variable_keys, variable_values


def convert_tuple2list(_dict: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in _dict.items():
        if isinstance(v, tuple):
            _dict[k] = list(v)

    return _dict


def main() -> None:
    args = get_arguments()

    # convert Namespace to dictionary.
    args_dict = vars(args).copy()
    del args_dict["root_dir"]

    base_config, variable_keys, variable_values = parse_params(args_dict)

    # get direct product
    product = itertools.product(*variable_values)

    # make a directory and save configuration file there.
    for values in product:
        config = base_config.copy()
        param_list = []
        for k, v in zip(variable_keys, values):
            config[k] = v
            param_list.append(f"{k}-{v}")

        # tuple should be saved as list
        config = convert_tuple2list(config)

        dir_name = "_".join(param_list)
        dir_path = os.path.join(args.root_dir, dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        config_path = os.path.join(dir_path, "config.yaml")

        # save configuration file as yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print("Finished making configuration files.")


if __name__ == "__main__":
    main()
