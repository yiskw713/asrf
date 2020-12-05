import argparse
import glob
import os

import pandas as pd


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="make csv files for training and testing."
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

    # create csv directory
    csv_dir = "./csv"
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    datasets = ["50salads", "gtea", "breakfast"]

    for dataset in datasets:
        # make directory for saving csv files
        save_dir = os.path.join(csv_dir, dataset)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        train_splits_paths = glob.glob(
            os.path.join(args.dataset_dir, dataset, "splits", "train.*")
        )
        test_splits_paths = glob.glob(
            os.path.join(args.dataset_dir, dataset, "splits", "test.*")
        )

        train_splits_paths = sorted(train_splits_paths)
        test_splits_paths = sorted(test_splits_paths)

        for i in range(len(train_splits_paths)):
            with open(train_splits_paths[i], "r") as f:
                train_ids = f.read().split("\n")[:-1]

            # remove .txt from file name of train_ids
            train_ids = [train_id[:-4] for train_id in train_ids]

            train_feature_paths = []
            train_label_paths = []
            train_boundary_paths = []
            val_feature_paths = []
            val_label_paths = []
            val_boundary_paths = []

            # split train and val
            for j in range(len(train_ids)):
                if j % 10 == 9:
                    val_feature_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, "features", train_ids[j] + ".npy"
                        )
                    )
                    val_label_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, "gt_arr", train_ids[j] + ".npy"
                        )
                    )
                    val_boundary_paths.append(
                        os.path.join(
                            args.dataset_dir,
                            dataset,
                            "gt_boundary_arr",
                            train_ids[j] + ".npy",
                        )
                    )
                else:
                    train_feature_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, "features", train_ids[j] + ".npy"
                        )
                    )
                    train_label_paths.append(
                        os.path.join(
                            args.dataset_dir, dataset, "gt_arr", train_ids[j] + ".npy"
                        )
                    )
                    train_boundary_paths.append(
                        os.path.join(
                            args.dataset_dir,
                            dataset,
                            "gt_boundary_arr",
                            train_ids[j] + ".npy",
                        )
                    )

            # test data list
            with open(test_splits_paths[i], "r") as f:
                test_ids = f.read().split("\n")[:-1]

            # remove .txt from file name of test_ids
            test_ids = [test_id[:-4] for test_id in test_ids]

            test_feature_paths = [
                os.path.join(args.dataset_dir, dataset, "features", test_id + ".npy")
                for test_id in test_ids
            ]
            test_label_paths = [
                os.path.join(args.dataset_dir, dataset, "gt_arr", test_id + ".npy")
                for test_id in test_ids
            ]
            test_boundary_paths = [
                os.path.join(
                    args.dataset_dir, dataset, "gt_boundary_arr", test_id + ".npy"
                )
                for test_id in test_ids
            ]

            # make dataframe to save csv files
            train_df = pd.DataFrame(
                {
                    "feature": train_feature_paths,
                    "label": train_label_paths,
                    "boundary": train_boundary_paths,
                },
                columns=["feature", "label", "boundary"],
            )

            val_df = pd.DataFrame(
                {
                    "feature": val_feature_paths,
                    "label": val_label_paths,
                    "boundary": val_boundary_paths,
                },
                columns=["feature", "label", "boundary"],
            )

            test_df = pd.DataFrame(
                {
                    "feature": test_feature_paths,
                    "label": test_label_paths,
                    "boundary": test_boundary_paths,
                },
                columns=["feature", "label", "boundary"],
            )

            train_df.to_csv(
                os.path.join(save_dir, "train{}.csv".format(i + 1)), index=None
            )
            val_df.to_csv(os.path.join(save_dir, "val{}.csv".format(i + 1)), index=None)
            test_df.to_csv(
                os.path.join(save_dir, "test{}.csv".format(i + 1)), index=None
            )

    print("Done")


if __name__ == "__main__":
    main()
