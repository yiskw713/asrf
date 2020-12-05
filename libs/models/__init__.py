from typing import Any

import torch.nn as nn

from .tcn import EDTCN, ActionSegmentRefinementFramework, MultiStageTCN, SingleStageTCN

__all__ = ["get_model", "ActionSegmentRefinementFramework"]

model_names = ["ms-tcn", "ss-tcn", "ed-tcn"]


def get_as_model(model_name: str, **kwargs: Any) -> nn.Module:
    model_name = model_name.lower()

    if model_name not in model_names:
        raise ValueError("Model name must be either 'ms-tcn', 'ss-tcn' or 'ed-tcn'.")

    if model_name == "ms-tcn":
        model = MultiStageTCN(**kwargs)
    elif model_name == "ss-tcn":
        model = SingleStageTCN(**kwargs)
    elif model_name == "ed-tcn":
        model = EDTCN(**kwargs)

    return model
