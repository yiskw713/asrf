import sys
from typing import Optional

import torch
import torch.nn as nn

from .focal import FocalLoss
from .tmse import TMSE, GaussianSimilarityTMSE

__all__ = ["ActionSegmentationLoss", "BoundaryRegressionLoss"]


class ActionSegmentationLoss(nn.Module):
    """
    Loss Function for Action Segmentation
    You can choose the below loss functions and combine them.
        - Cross Entropy Loss (CE)
        - Focal Loss
        - Temporal MSE (TMSE)
        - Gaussian Similarity TMSE (GSTMSE)
    """

    def __init__(
        self,
        ce: bool = True,
        focal: bool = True,
        tmse: bool = False,
        gstmse: bool = False,
        weight: Optional[float] = None,
        threshold: float = 4,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        focal_weight: float = 1.0,
        tmse_weight: float = 0.15,
        gstmse_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.criterions = []
        self.weights = []

        if ce:
            self.criterions.append(
                nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            )
            self.weights.append(ce_weight)

        if focal:
            self.criterions.append(FocalLoss(ignore_index=ignore_index))
            self.weights.append(focal_weight)

        if tmse:
            self.criterions.append(TMSE(threshold=threshold, ignore_index=ignore_index))
            self.weights.append(tmse_weight)

        if gstmse:
            self.criterions.append(
                GaussianSimilarityTMSE(threshold=threshold, ignore_index=ignore_index)
            )
            self.weights.append(gstmse_weight)

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def forward(
        self, preds: torch.Tensor, gts: torch.Tensor, sim_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds: torch.float (N, C, T).
            gts: torch.long (N, T).
            sim_index: torch.float (N, C', T).
        """

        loss = 0.0
        for criterion, weight in zip(self.criterions, self.weights):
            if isinstance(criterion, GaussianSimilarityTMSE):
                loss += weight * criterion(preds, gts, sim_index)
            else:
                loss += weight * criterion(preds, gts)

        return loss


class BoundaryRegressionLoss(nn.Module):
    """
    Boundary Regression Loss
        bce: Binary Cross Entropy Loss for Boundary Prediction
        mse: Mean Squared Error
    """

    def __init__(
        self,
        bce: bool = True,
        focal: bool = False,
        mse: bool = False,
        weight: Optional[float] = None,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.criterions = []

        if bce:
            self.criterions.append(
                nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
            )

        if focal:
            self.criterions.append(FocalLoss())

        if mse:
            self.criterions.append(nn.MSELoss())

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def forward(self, preds: torch.Tensor, gts: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            preds: torch.float (N, 1, T).
            gts: torch. (N, 1, T).
            masks: torch.bool (N, 1, T).
        """
        loss = 0.0
        batch_size = float(preds.shape[0])

        for criterion in self.criterions:
            for pred, gt, mask in zip(preds, gts, masks):
                loss += criterion(pred[mask], gt[mask])

        return loss / batch_size
