import dataclasses
import pprint
from typing import Any, Dict, Tuple

import yaml

__all__ = ["get_config"]


@dataclasses.dataclass
class Config:
    model: str = "ActionSegmentRefinementNetwork"
    n_layers: int = 10
    n_stages: int = 4  # for ms-tcn
    n_features: int = 64
    n_stages_asb: int = 4
    n_stages_brb: int = 4

    # loss function
    ce: bool = True  # cross entropy
    ce_weight: float = 1.0

    focal: bool = False
    focal_weight: float = 1.0

    tmse: bool = False  # temporal mse
    tmse_weight: float = 0.15

    gstmse: bool = True  # gaussian similarity loss
    gstmse_weight: float = 1.0
    gstmse_index: str = "feature"  # similarity index

    # if you use class weight to calculate cross entropy or not
    class_weight: bool = True

    batch_size: int = 1

    # the number of input feature channels
    in_channel: int = 2048

    num_workers: int = 4
    max_epoch: int = 50

    optimizer: str = "Adam"

    learning_rate: float = 0.0005
    momentum: float = 0.9  # momentum of SGD
    dampening: float = 0.0  # dampening for momentum of SGD
    weight_decay: float = 0.0001  # weight decay
    nesterov: bool = True  # enables Nesterov momentum

    param_search: bool = False

    # thresholds for calcualting F1 Score
    iou_thresholds: Tuple[float, ...] = (0.1, 0.25, 0.5)

    # boundary regression
    tolerance: int = 5
    boundary_th: float = 0.5
    lambda_b: float = 0.1

    dataset: str = "breakfast"
    dataset_dir: str = "./dataset"
    csv_dir: str = "./csv"
    split: int = 1

    def __post_init__(self) -> None:
        self._type_check()

        print("-" * 10, "Experiment Configuration", "-" * 10)
        pprint.pprint(dataclasses.asdict(self), width=1)

    def _type_check(self) -> None:
        """Reference:
        https://qiita.com/obithree/items/1c2b43ca94e4fbc3aa8d
        """

        _dict = dataclasses.asdict(self)

        for field, field_type in self.__annotations__.items():
            # if you use type annotation class provided by `typing`,
            # you should convert it to the type class used in python.
            # e.g.) Tuple[int] -> tuple
            # https://stackoverflow.com/questions/51171908/extracting-data-from-typing-types

            # check the instance is Tuple or not.
            # https://github.com/zalando/connexion/issues/739
            if hasattr(field_type, "__origin__"):
                # e.g.) Tuple[int].__args__[0] -> `int`
                element_type = field_type.__args__[0]

                # e.g.) Tuple[int].__origin__ -> `tuple`
                field_type = field_type.__origin__

                self._type_check_element(field, _dict[field], element_type)

            # bool is the subclass of int,
            # so need to use `type() is` instead of `isinstance`
            if type(_dict[field]) is not field_type:
                raise TypeError(
                    f"The type of '{field}' field is supposed to be {field_type}."
                )

    def _type_check_element(
        self, field: str, vals: Tuple[Any], element_type: type
    ) -> None:
        for val in vals:
            if type(val) is not element_type:
                raise TypeError(
                    f"The element of '{field}' field is supposed to be {element_type}."
                )


def convert_list2tuple(_dict: Dict[str, Any]) -> Dict[str, Any]:
    # cannot use list in dataclass because mutable defaults are not allowed.
    for key, val in _dict.items():
        if isinstance(val, list):
            _dict[key] = tuple(val)

    return _dict


def get_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict = convert_list2tuple(config_dict)
    config = Config(**config_dict)
    return config
