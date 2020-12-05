import torch
import torch.nn as nn
import torch.nn.functional as F


class TMSE(nn.Module):
    """
    Temporal MSE Loss Function
    Proposed in Y. A. Farha et al. MS-TCN: Multi-Stage Temporal Convolutional Network for ActionSegmentation in CVPR2019
    arXiv: https://arxiv.org/pdf/1903.01945.pdf
    """

    def __init__(self, threshold: float = 4, ignore_index: int = 255) -> None:
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:

        total_loss = 0.0
        batch_size = preds.shape[0]
        for pred, gt in zip(preds, gts):
            pred = pred[:, torch.where(gt != self.ignore_index)[0]]

            loss = self.mse(
                F.log_softmax(pred[:, 1:], dim=1), F.log_softmax(pred[:, :-1], dim=1)
            )

            loss = torch.clamp(loss, min=0, max=self.threshold ** 2)
            total_loss += torch.mean(loss)

        return total_loss / batch_size


class GaussianSimilarityTMSE(nn.Module):
    """
    Temporal MSE Loss Function with Gaussian Similarity Weighting
    """

    def __init__(
        self, threshold: float = 4, sigma: float = 1.0, ignore_index: int = 255
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")
        self.sigma = sigma

    def forward(
        self, preds: torch.Tensor, gts: torch.Tensor, sim_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds: the output of model before softmax. (N, C, T)
            gts: Ground Truth. (N, T)
            sim_index: similarity index. (N, C, T)
        Return:
            the value of Temporal MSE weighted by Gaussian Similarity.
        """
        total_loss = 0.0
        batch_size = preds.shape[0]
        for pred, gt, sim in zip(preds, gts, sim_index):
            pred = pred[:, torch.where(gt != self.ignore_index)[0]]
            sim = sim[:, torch.where(gt != self.ignore_index)[0]]

            # calculate gaussian similarity
            diff = sim[:, 1:] - sim[:, :-1]
            similarity = torch.exp(-torch.norm(diff, dim=0) / (2 * self.sigma ** 2))

            # calculate temporal mse
            loss = self.mse(
                F.log_softmax(pred[:, 1:], dim=1), F.log_softmax(pred[:, :-1], dim=1)
            )
            loss = torch.clamp(loss, min=0, max=self.threshold ** 2)

            # gaussian similarity weighting
            loss = similarity * loss

            total_loss += torch.mean(loss)

        return total_loss / batch_size
