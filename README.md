# Alleviating Over-segmentation Errors by Detecting Action Boundaries

This repo is the official implementation of [Y. Ishikawa et al. "Alleviating Over-segmentation Errors by Detecting Action Boundaries" in WACV 2021](https://arxiv.org/abs/2007.06866).

## Dataset

GTEA, 50Salads, Breakfast  

You can download features and G.T. of these datasets from [this repository](https://github.com/yabufarha/ms-tcn).  
Or you can extract their features by yourself using [this repository](https://github.com/yiskw713/video_feature_extractor)

## Requirements

* Python >= 3.7
* pytorch => 1.0
* torchvision
* pandas
* numpy
* Pillow
* PyYAML

You can download packages using requirements.txt.  

```bash
pip install -r requirements.txt
```

## Directory Structure

```directory structure
root ── csv/
      ├─ libs/
      ├─ imgs/
      ├─ result/
      ├─ utils/
      ├─ dataset ─── 50salads/...
      │           ├─ breakfast/...
      │           └─ gtea ─── features/
      │                    ├─ groundTruth/
      │                    ├─ splits/
      │                    └─ mapping.txt
      ├.gitignore
      ├ README.md
      ├ requirements.txt
      ├ save_pred.py
      ├ train.py
      └ evaluate.py
```

* `csv` directory contains csv files which are necessary for training and testing.
* An image in `imgs` is one from PascalVOC. This is used for an color palette to visualize outputs.
* Experimental results are stored in `results` directory.
* Scripts in `utils` are directly irrelevant with `train.py` and `evaluate.py` but necessary for converting labels, generating configurations, visualization and so on.
* Scripts in `libs` are necessary for training and evaluation. e.g.) models, loss functions, dataset class and so on.
* The datasets downloaded from [this repository](https://github.com/yabufarha/ms-tcn) are stored in `dataset`.
  You can put them in another directory, but need to specify the path in configuration files.
* `train.py` is a script for training networks.
* `eval.py` is a script for evaluation.
* `save_pred.py` is for saving predictions from models.

## How to use

Please also check `scripts/experiment.sh`, which runs all the following experimental codes.

1. First of all, please download features and G.T. of these datasets from [this repository](https://github.com/yabufarha/ms-tcn).

1. Features and groundTruth labels need to be converted to numpy array. [This repository](https://github.com/yabufarha/ms-tcn) does not provide boundary groundtruth labels, so you have to generate them, too.
Please run the following command. `[DATASET_DIR]` is the path to your dataset directory.

    ```bash
    python utils/generate_gt_array.py --dataset_dir [DATASET_DIR]
    python utils/generate_boundary_array.py --dataset_dir [DATASET_DIR]
    ```

1. In this implementation, csv files are used for keeping information  of training or test data. Please run the below command to generate csv files.

    ```bash
    python utils/make_csv_files.py --dataset_dir [DATASET_DIR]
    ```

1. You can automatically generate experiment configuration files by running the following command. This command generates directories and configuration files in `root_dir`.

    ```bash
    python utils/make_config.py --root_dir ./result/50salads --dataset 50salads --split 1 2 3 4 5
    python utils/make_config.py --root_dir ./result/gtea --dataset gtea --split 1 2 3 4
    python utils/make_config.py --root_dir ./result/breakfast --dataset breakfast --split 1 2 3 4
    ```

    If you want to add other configurations, please add command-line options like:

    ```bash
    python utils/make_config.py --root_dir ./result/50salads --dataset 50salads --split 1 2 3 4 5 --learning_rate 0.1 0.01 0.001 0.0001
    ```

    Please see `libs/config.py` about configurations.

1. You can train and evaluate models specifying a configuration file generated in the above process like:

    ```bash
    python train.py ./result/50salads/dataset-50salads_split-1/config.yaml
    python evaluate.py ./result/50salads/dataset-50salads_split-1/config.yaml test
    ```

1. You can also save model predictions as numpy array by running:

    ```bash
    python save_pred.py ./result/50salads/dataset-50salads_split-1/config.yaml test
    ```

1. If you want to visualize the saved model predictions, please run:

    ```bash
    python utils/convert_arr2img.py ./result/50salads/dataset-50salads_split1/predictions
    ```

## License

This repository is released under the MIT License.

## Citation

```citation
Yuchi Ishikawa, Seito Kasai, Yoshimitsu Aoki, Hirokatsu Kataoka,
"Alleviating Over-segmentation Errors by Detecting Action Boundaries"
in WACV 2021
```

You can see the paper in [arXiv](https://arxiv.org/abs/2007.06866)

## Reference

* Colin Lea et al., "Temporal Convolutional Networks for Action Segmentation and Detection", in CVPR2017 ([paper](http://zpascal.net/cvpr2017/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.pdf))
* Yazan Abu Farha et al., "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation", in CVPR2019 ([paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.pdf), [code](https://github.com/yabufarha/ms-tcn))
