import argparse
import glob
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from libs.class_id_map import get_class2id_map


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="convert ground truth txt files to numpy array"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="path to a dataset directory (default: ./dataset)",
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    datasets = ["50salads", "gtea", "breakfast"]

    for dataset in datasets:
        # make directory for saving ground truth numpy arrays
        save_dir = os.path.join(args.dataset_dir, dataset, "gt_arr")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # class to index mapping
        class2id_map = get_class2id_map(dataset, dataset_dir=args.dataset_dir)

        gt_dir = os.path.join(args.dataset_dir, dataset, "groundTruth")
        gt_paths = glob.glob(os.path.join(gt_dir, "*.txt"))

        for gt_path in gt_paths:
            # the name of ground truth text file
            gt_name = os.path.relpath(gt_path, gt_dir)

            with open(gt_path, "r") as f:
                gt = f.read().split("\n")[:-1]

            gt_array = np.zeros(len(gt))
            for i in range(len(gt)):
                gt_array[i] = class2id_map[gt[i]]

            # save array
            np.save(os.path.join(save_dir, gt_name[:-4] + ".npy"), gt_array)

    print("Done")


if __name__ == "__main__":
    main()
