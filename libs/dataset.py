import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose

__all__ = ["ActionSegmentationDataset", "collate_fn"]

dataset_names = ["50salads", "breakfast", "gtea"]
modes = ["training", "validation", "trainval", "test"]


class ActionSegmentationDataset(Dataset):
    """ Action Segmentation Dataset (50salads, gtea, breakfast) """

    def __init__(
        self,
        dataset: str,
        transform: Optional[Compose] = None,
        mode: str = "training",
        split: int = 1,
        dataset_dir: str = "./dataset",
        csv_dir: str = "./csv",
    ) -> None:
        super().__init__()
        """
            Args:
                dataset: the name of dataset (50salads, gtea, breakfast)
                transform: torchvision.transforms.Compose([...])
                mode: training, validation, test
                split: which split of train, val and test do you want to use in csv files.(default:1)
                csv_dir: the path to the directory where the csv files are saved
        """

        assert (
            dataset in dataset_names
        ), "You have to choose 50saladas, gtea, breakfast as dataset."

        if mode == "training":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            )
        elif mode == "validation":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "val{}.csv".format(split))
            )
        elif mode == "trainval":
            df1 = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            )
            df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
            self.df = pd.concat([df1, df2])
        elif mode == "test":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "test{}.csv".format(split))
            )
        else:
            assert (
                mode in modes
            ), "You have to choose 'training', 'trainval', 'validation' or 'test' as the dataset mode."

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feature_path = self.df.iloc[idx]["feature"]
        label_path = self.df.iloc[idx]["label"]
        boundary_path = self.df.iloc[idx]["boundary"]

        feature = np.load(feature_path).astype(np.float32)
        label = np.load(label_path).astype(np.int64)
        boundary = np.load(boundary_path).astype(np.float32)

        if self.transform is not None:
            feature, label, boundary = self.transform([feature, label, boundary])

        sample = {
            "feature": feature,
            "label": label,
            "feature_path": feature_path,
            "boundary": boundary,
        }

        return sample


def collate_fn(sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_length = max([s["feature"].shape[1] for s in sample])

    feat_list = []
    label_list = []
    path_list = []
    boundary_list = []
    length_list = []

    for s in sample:
        feature = s["feature"]
        label = s["label"]
        boundary = s["boundary"]
        feature_path = s["feature_path"]

        _, t = feature.shape
        pad_t = max_length - t

        length_list.append(t)

        if pad_t > 0:
            feature = F.pad(feature, (0, pad_t), mode="constant", value=0.0)
            label = F.pad(label, (0, pad_t), mode="constant", value=255)
            boundary = F.pad(boundary, (0, pad_t), mode="constant", value=0.0)

        # reshape boundary (T) => (1, T)
        boundary = boundary.unsqueeze(0)

        feat_list.append(feature)
        label_list.append(label)
        path_list.append(feature_path)
        boundary_list.append(boundary)

    # merge features from tuple of 2D tensor to 3D tensor
    features = torch.stack(feat_list, dim=0)
    # merge labels from tuple of 1D tensor to 2D tensor
    labels = torch.stack(label_list, dim=0)

    # merge labels from tuple of 2D tensor to 3D tensor
    # shape (N, 1, T)
    boundaries = torch.stack(boundary_list, dim=0)

    # generate masks which shows valid length for each video (N, 1, T)
    masks = [
        [[1 if i < length else 0 for i in range(max_length)]] for length in length_list
    ]
    masks = torch.tensor(masks, dtype=torch.bool)

    return {
        "feature": features,
        "label": labels,
        "boundary": boundaries,
        "feature_path": path_list,
        "mask": masks,
    }
