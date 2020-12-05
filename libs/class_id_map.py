import os
from typing import Dict

__all__ = ["get_class2id_map", "get_id2class_map", "get_n_classes"]

dataset_names = ["50salads", "breakfast", "gtea"]


def get_class2id_map(dataset: str, dataset_dir: str = "./dataset") -> Dict[str, int]:
    """
    Args:
        dataset: 50salads, gtea, breakfast
        dataset_dir: the path to the datset directory
    """

    assert (
        dataset in dataset_names
    ), "You have to choose 50salads, gtea or breakfast as dataset."

    with open(os.path.join(dataset_dir, "{}/mapping.txt".format(dataset)), "r") as f:
        actions = f.read().split("\n")[:-1]

    class2id_map = dict()
    for a in actions:
        class2id_map[a.split()[1]] = int(a.split()[0])

    return class2id_map


def get_id2class_map(dataset: str, dataset_dir: str = "./dataset") -> Dict[int, str]:
    class2id_map = get_class2id_map(dataset, dataset_dir)

    return {val: key for key, val in class2id_map.items()}


def get_n_classes(dataset: str, dataset_dir: str = "./dataset") -> int:
    return len(get_class2id_map(dataset, dataset_dir))
