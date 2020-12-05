import os
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_loss: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    torch.save(save_states, os.path.join(result_path, "checkpoint.pth"))


def resume(
    result_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[Any]:

    resume_path = os.path.join(result_path, "checkpoint.pth")
    print("loading checkpoint {}".format(resume_path))

    checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)

    begin_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])

    # confirm whether the optimizer matches that of checkpoints
    optimizer.load_state_dict(checkpoint["optimizer"])

    return begin_epoch, model, optimizer, best_loss
