import argparse
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.checkpoint import resume, save_checkpoint
from libs.class_id_map import get_n_classes
from libs.class_weight import get_class_weight, get_pos_weight
from libs.config import get_config
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.helper import train, validate
from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss
from libs.optimizer import get_optimizer
from libs.transformer import TempDownSamp, ToTensor


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="train a network for action recognition"
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="a number used to initialize a pseudorandom number generator.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )

    return parser.parse_args()


def main() -> None:
    # argparser
    args = get_arguments()

    # configuration
    config = get_config(args.config)

    result_path = os.path.dirname(args.config)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # cpu or cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Dataloader
    # Temporal downsampling is applied to only videos in 50Salads
    downsamp_rate = 2 if config.dataset == "50salads" else 1

    train_data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
        mode="trainval" if not config.param_search else "training",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True if config.batch_size > 1 else False,
        collate_fn=collate_fn,
    )

    # if you do validation to determine hyperparams
    if config.param_search:
        val_data = ActionSegmentationDataset(
            config.dataset,
            transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
            mode="validation",
            split=config.split,
            dataset_dir=config.dataset_dir,
            csv_dir=config.csv_dir,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

    # load model
    print("---------- Loading Model ----------")

    n_classes = get_n_classes(config.dataset, dataset_dir=config.dataset_dir)

    model = models.ActionSegmentRefinementFramework(
        in_channel=config.in_channel,
        n_features=config.n_features,
        n_classes=n_classes,
        n_stages=config.n_stages,
        n_layers=config.n_layers,
        n_stages_asb=config.n_stages_asb,
        n_stages_brb=config.n_stages_brb,
    )

    # send the model to cuda/cpu
    model.to(device)

    optimizer = get_optimizer(
        config.optimizer,
        model,
        config.learning_rate,
        momentum=config.momentum,
        dampening=config.dampening,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )

    # resume if you want
    columns = ["epoch", "lr", "train_loss"]

    # if you do validation to determine hyperparams
    if config.param_search:
        columns += ["val_loss", "cls_acc", "edit"]
        columns += [
            "segment f1s@{}".format(config.iou_thresholds[i])
            for i in range(len(config.iou_thresholds))
        ]
        columns += ["bound_acc", "precision", "recall", "bound_f1s"]

    begin_epoch = 0
    best_loss = float("inf")
    log = pd.DataFrame(columns=columns)

    if args.resume:
        if os.path.exists(os.path.join(result_path, "checkpoint.pth")):
            checkpoint = resume(result_path, model, optimizer)
            begin_epoch, model, optimizer, best_loss = checkpoint
            log = pd.read_csv(os.path.join(result_path, "log.csv"))
            print("training will start from {} epoch".format(begin_epoch))
        else:
            print("there is no checkpoint at the result folder")

    # criterion for loss
    if config.class_weight:
        class_weight = get_class_weight(
            config.dataset,
            split=config.split,
            dataset_dir=config.dataset_dir,
            csv_dir=config.csv_dir,
            mode="training" if config.param_search else "trainval",
        )
        class_weight = class_weight.to(device)
    else:
        class_weight = None

    criterion_cls = ActionSegmentationLoss(
        ce=config.ce,
        focal=config.focal,
        tmse=config.tmse,
        gstmse=config.gstmse,
        weight=class_weight,
        ignore_index=255,
        ce_weight=config.ce_weight,
        focal_weight=config.focal_weight,
        tmse_weight=config.tmse_weight,
        gstmse_weight=config.gstmse,
    )

    pos_weight = get_pos_weight(
        dataset=config.dataset,
        split=config.split,
        csv_dir=config.csv_dir,
        mode="training" if config.param_search else "trainval",
    ).to(device)

    criterion_bound = BoundaryRegressionLoss(pos_weight=pos_weight)

    # train and validate model
    print("---------- Start training ----------")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        train_loss = train(
            train_loader,
            model,
            criterion_cls,
            criterion_bound,
            config.lambda_b,
            optimizer,
            epoch,
            device,
        )

        # if you do validation to determine hyperparams
        if config.param_search:
            (
                val_loss,
                cls_acc,
                edit_score,
                segment_f1s,
                bound_acc,
                precision,
                recall,
                bound_f1s,
            ) = validate(
                val_loader,
                model,
                criterion_cls,
                criterion_bound,
                config.lambda_b,
                device,
                config.dataset,
                config.dataset_dir,
                config.iou_thresholds,
                config.boundary_th,
                config.tolerance,
            )

            # save a model if top1 acc is higher than ever
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(result_path, "best_loss_model.prm"),
                )

        # save checkpoint every epoch
        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        # write logs to dataframe and csv file
        tmp = [epoch, optimizer.param_groups[0]["lr"], train_loss]

        # if you do validation to determine hyperparams
        if config.param_search:
            tmp += [
                val_loss,
                cls_acc,
                edit_score,
            ]
            tmp += segment_f1s
            tmp += [
                bound_acc,
                precision,
                recall,
                bound_f1s,
            ]

        tmp_df = pd.Series(tmp, index=log.columns)

        log = log.append(tmp_df, ignore_index=True)
        log.to_csv(os.path.join(result_path, "log.csv"), index=False)

        if config.param_search:
            # if you do validation to determine hyperparams
            print(
                "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}\tval loss: {:.4f}\tval_acc: {:.4f}\tedit: {:.4f}".format(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    train_loss,
                    val_loss,
                    cls_acc,
                    edit_score,
                )
            )
        else:
            print(
                "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}".format(
                    epoch, optimizer.param_groups[0]["lr"], train_loss
                )
            )

    # delete checkpoint
    os.remove(os.path.join(result_path, "checkpoint.pth"))

    # save models
    torch.save(model.state_dict(), os.path.join(result_path, "final_model.prm"))

    print("Done!")


if __name__ == "__main__":
    main()
