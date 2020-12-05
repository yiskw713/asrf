import torch.nn as nn
import torch.optim as optim


def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    momentum: float = 0.9,
    dampening: float = 0.0,
    weight_decay: float = 0.0001,
    nesterov: bool = True,
) -> optim.Optimizer:

    assert optimizer_name in ["SGD", "Adam"]
    print(f"{optimizer_name} will be used as an optimizer.")

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    return optimizer
