import argparse
import glob
import os

import numpy as np
import pandas as pd


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="average cross validation results.")
    parser.add_argument("result_dir", type=str, help="path to a result directory")
    parser.add_argument("--mode", type=str, default="test", help="[test or validation]")

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    sub_dirs = glob.glob(os.path.join(args.result_dir, "*"))

    values = []
    for sub_dir in sub_dirs:
        log_path = os.path.join(sub_dir, "{}_log.csv".format(args.mode))

        if not os.path.exists(log_path):
            continue

        df = pd.read_csv(log_path)
        values.append(df.values.tolist())

    values = np.mean(values, axis=0)
    values = pd.DataFrame(values, columns=df.columns)
    values.to_csv(
        os.path.join(args.result_dir, "average_{}_log.csv".format(args.mode)),
        index=False,
    )


if __name__ == "__main__":
    main()
