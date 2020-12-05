# FIRST OF ALL, YOU NEED TO DOWNLOAD THE FEATURES OF THE THREE DATASET FROM:
# https://github.com/yabufarha/ms-tcn

# create gt arrays
python utils/generate_gt_array.py --dataset_dir [DATASET_DIR]
python utils/generate_boundary_array.py --dataset_dir [DATASET_DIR]

# make csv files for training and testing
python utils/make_csv_files.py --dataset_dir [DATASET_DIR]

# make configuration files
python utils/make_config.py --root_dir ./result/50salads --dataset 50salads --split 1 2 3 4 5
python utils/make_config.py --root_dir ./result/gtea --dataset gtea --split 1 2 3 4
python utils/make_config.py --root_dir ./result/breakfast --dataset breakfast --split 1 2 3 4

# 50salads dataset
# training
python train.py ./result/50salads/dataset-50salads_split-1/config.yaml
python train.py ./result/50salads/dataset-50salads_split-2/config.yaml
python train.py ./result/50salads/dataset-50salads_split-3/config.yaml
python train.py ./result/50salads/dataset-50salads_split-4/config.yaml
python train.py ./result/50salads/dataset-50salads_split-5/config.yaml

# test
python evaluate.py ./result/50salads/dataset-50salads_split-1/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/50salads/dataset-50salads_split-2/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/50salads/dataset-50salads_split-3/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/50salads/dataset-50salads_split-4/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/50salads/dataset-50salads_split-5/config.yaml test --refinement_method refinement_with_boundary

# gtea dataset
# training
python train.py ./result/gtea/dataset-gtea_split-1/config.yaml
python train.py ./result/gtea/dataset-gtea_split-2/config.yaml
python train.py ./result/gtea/dataset-gtea_split-3/config.yaml
python train.py ./result/gtea/dataset-gtea_split-4/config.yaml

# test
python evaluate.py ./result/gtea/dataset-gtea_split-1/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/gtea/dataset-gtea_split-2/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/gtea/dataset-gtea_split-3/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/gtea/dataset-gtea_split-4/config.yaml test --refinement_method refinement_with_boundary

# breakfast dataset
# training
python train.py ./result/breakfast/dataset-breakfast_split-1/config.yaml
python train.py ./result/breakfast/dataset-breakfast_split-2/config.yaml
python train.py ./result/breakfast/dataset-breakfast_split-3/config.yaml
python train.py ./result/breakfast/dataset-breakfast_split-4/config.yaml

# test
python evaluate.py ./result/breakfast/dataset-breakfast_split-1/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/breakfast/dataset-breakfast_split-2/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/breakfast/dataset-breakfast_split-3/config.yaml test --refinement_method refinement_with_boundary
python evaluate.py ./result/breakfast/dataset-breakfast_split-4/config.yaml test --refinement_method refinement_with_boundary

# average cross validation results.
python utils/average_cv_results.py ./result/50salads
python utils/average_cv_results.py ./result/gtea
python utils/average_cv_results.py ./result/breakfast

# output visualization
# save outputs as np.array
python save_pred.py ./result/50salads/dataset-50salads_split1/config.yaml test --refinement_method refinement_with_boundary

# convert array to image
python utils/convert_arr2img.py ./result/50salads/dataset-50salads_split1/predictions
