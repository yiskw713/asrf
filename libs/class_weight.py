import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from libs.class_id_map import get_n_classes

__all__ = ["get_pos_weight", "get_class_weight"]

dataset_names = ["50salads", "breakfast", "gtea"]
modes = ["training", "trainval"]


def get_class_nums(
    dataset: str,
    split: int = 1,
    dataset_dir: str = "./dataset/",
    csv_dir: str = "./csv",
    mode: str = "trainval",
) -> List[int]:

    assert (
        mode in modes
    ), "You have to choose 'training' or 'trainval' as the dataset mode."

    if mode == "training":
        df = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv").format(split))
    elif mode == "trainval":
        df1 = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv".format(split)))
        df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
        df = pd.concat([df1, df2])

    assert (
        dataset in dataset_names
    ), "You have to select 50salads, gtea or breakfast as dataset."

    n_classes = get_n_classes(dataset, dataset_dir)

    nums = [0 for i in range(n_classes)]
    for i in range(len(df)):
        label_path = df.iloc[i]["label"]
        label = np.load(label_path).astype(np.int64)
        num, cnt = np.unique(label, return_counts=True)
        for n, c in zip(num, cnt):
            nums[n] += c

    return nums


def get_class_weight(
    dataset: str,
    split: int = 1,
    dataset_dir: str = "./dataset",
    csv_dir: str = "./csv",
    mode: str = "trainval",
) -> torch.Tensor:
    """
    Class weight for CrossEntropy
    Class weight is calculated in the way described in:
        D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in ICCV,
        openaccess: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
    """

    nums = get_class_nums(dataset, split, dataset_dir, csv_dir, mode)

    class_num = torch.tensor(nums)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency

    return class_weight


def get_pos_weight(
    dataset: str,
    split: int = 1,
    csv_dir: str = "./csv",
    mode: str = "trainval",
    norm: Optional[float] = None,
) -> torch.Tensor:
    """
    pos_weight for binary cross entropy with logits loss
    pos_weight is defined as reciprocal of ratio of positive samples in the dataset
    """

    assert (
        mode in modes
    ), "You have to choose 'training' or 'trainval' as the dataset mode"

    assert (
        dataset in dataset_names
    ), "You have to select 50salads, gtea or breakfast as dataset."

    if mode == "training":
        df = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv").format(split))
    elif mode == "trainval":
        df1 = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv".format(split)))
        df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
        df = pd.concat([df1, df2])

    n_classes = 2  # boundary or not
    nums = [0 for i in range(n_classes)]
    for i in range(len(df)):
        label_path = df.iloc[i]["boundary"]
        label = np.load(label_path).astype(np.int64)
        num, cnt = np.unique(label, return_counts=True)
        for n, c in zip(num, cnt):
            nums[n] += c

    pos_ratio = nums[1] / sum(nums)
    pos_weight = 1 / pos_ratio

    if norm is not None:
        pos_weight /= norm

    return torch.tensor(pos_weight)
