# Originally written by yabufarha
# https://github.com/yabufarha/ms-tcn/blob/master/model.py

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStageTCN(nn.Module):
    """
    Y. Abu Farha and J. Gall.
    MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

    parameters used in originl paper:
        n_features: 64
        n_stages: 4
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.stage1 = SingleStageTCN(in_channel, n_features, n_classes, n_layers)

        stages = [
            SingleStageTCN(n_classes, n_features, n_classes, n_layers)
            for _ in range(n_stages - 1)
        ]
        self.stages = nn.ModuleList(stages)

        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # for training
            outputs = []
            out = self.stage1(x)
            outputs.append(out)
            for stage in self.stages:
                out = stage(self.activation(out))
                outputs.append(out)
            return outputs
        else:
            # for evaluation
            out = self.stage1(x)
            for stage in self.stages:
                out = stage(self.activation(out))
            return out


class SingleStageTCN(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channel: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out


class NormalizedReLU(nn.Module):
    """
    Normalized ReLU Activation prposed in the original TCN paper.
    the values are divided by the max computed per frame
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x /= x.max(dim=1, keepdim=True)[0] + self.eps

        return x


class EDTCN(nn.Module):
    """
    Encoder Decoder Temporal Convolutional Network
    """

    def __init__(
        self,
        in_channel: int,
        n_classes: int,
        kernel_size: int = 25,
        mid_channels: Tuple[int, int] = [128, 160],
        **kwargs: Any
    ) -> None:
        """
        Args:
            in_channel: int. the number of the channels of input feature
            n_classes: int. output classes
            kernel_size: int. 25 is proposed in the original paper
            mid_channels: list. the list of the number of the channels of the middle layer.
                        [96 + 32*1, 96 + 32*2] is proposed in the original paper
        Note that this implementation only supports n_layer=2
        """
        super().__init__()

        # encoder
        self.enc1 = nn.Conv1d(
            in_channel,
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = NormalizedReLU()

        self.enc2 = nn.Conv1d(
            mid_channels[0],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = NormalizedReLU()

        # decoder
        self.dec1 = nn.Conv1d(
            mid_channels[1],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout3 = nn.Dropout(0.3)
        self.relu3 = NormalizedReLU()

        self.dec2 = nn.Conv1d(
            mid_channels[1],
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout4 = nn.Dropout(0.3)
        self.relu4 = NormalizedReLU()

        self.conv_out = nn.Conv1d(mid_channels[0], n_classes, 1, bias=True)

        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder 1
        x1 = self.relu1(self.dropout1(self.enc1(x)))
        t1 = x1.shape[2]
        x1 = F.max_pool1d(x1, 2)

        # encoder 2
        x2 = self.relu2(self.dropout2(self.enc2(x1)))
        t2 = x2.shape[2]
        x2 = F.max_pool1d(x2, 2)

        # decoder 1
        x3 = F.interpolate(x2, size=(t2,), mode="nearest")
        x3 = self.relu3(self.dropout3(self.dec1(x3)))

        # decoder 2
        x4 = F.interpolate(x3, size=(t1,), mode="nearest")
        x4 = self.relu4(self.dropout4(self.dec2(x4)))

        out = self.conv_out(x4)

        return out

    def init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class ActionSegmentRefinementFramework(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 2048
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        shared_layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.shared_layers = nn.ModuleList(shared_layers)
        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)

        # action segmentation branch
        asb = [
            SingleStageTCN(n_classes, n_features, n_classes, n_layers)
            for _ in range(n_stages_asb - 1)
        ]

        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.conv_in(x)
        for layer in self.shared_layers:
            out = layer(out)

        out_cls = self.conv_cls(out)
        out_bound = self.conv_bound(out)

        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]

            for as_stage in self.asb:
                out_cls = as_stage(self.activation_asb(out_cls))
                outputs_cls.append(out_cls)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound))
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound)
        else:
            for as_stage in self.asb:
                out_cls = as_stage(self.activation_asb(out_cls))

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound))

            return (out_cls, out_bound)
