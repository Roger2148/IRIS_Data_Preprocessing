"""
This project is by Heng Zhang, Kyushu University, Japan.
"""

import os
import time
import numpy as np
import warnings

from utils.utils_iris_dataset_handler import IrisDatasetHandler

################################################################################
# config path
current_path = os.getcwd()
# check if data/iris_data folder exists, if not, create one
if not os.path.exists(current_path + '/data/iris_data'):
    os.makedirs(current_path + '/data/iris_data')
save_path = current_path + '/data/iris_data'
save_with_timestamp = True

# create a folder with timestamp as the folder name
if save_with_timestamp:
    time.sleep(1)  # make sure the timestamp is different
    timestamp = time.strftime("%Y%m%d%H%M%S")
    folder_name = timestamp
    if not os.path.exists(os.path.join(save_path, folder_name)):
        os.makedirs(os.path.join(save_path, folder_name))
else:
    folder_name = 'backup'
    if not os.path.exists(os.path.join(save_path, folder_name)):
        os.makedirs(os.path.join(save_path, folder_name))
    # check if there are files in the backup_reference folder
    if len(os.listdir(os.path.join(save_path, folder_name))) > 0:
        raise ValueError(f"Backup folder {os.path.join(save_path, folder_name)} is not empty, please delete the data first.")

################################################################################

# random_state = np.random.randint(1000)
random_state = 42 # this is a global random state for all the random operations in this script (will pass to dataset handler)
# warnings the random state
warnings.warn(f"Random state is set to {random_state} for all the random operations in this script for reproducibility. If you want to change the random state, please change the random_state parameter in the script.")
if save_with_timestamp == False:
    warnings.warn(f"Warning: saving data to /backup_reference folder. If you want to save data to a new folder with timestamp, please change the save_with_timestamp parameter in the script.")
time.sleep(3)


################################################################################
# initialize parameters
K_fold = 30
test_size = 0.5 # 0.3
normalize = False
duplicate = 3
noise_type = "test"
noise_SNR = 10
# generate a list of random state number for K-fold cross validation by random.choice()
rng = np.random.RandomState(random_state)
random_state_list = rng.choice(1000, K_fold, replace=False)

# print out parameters
print("[IRIS PREPROCESSING] Parameters:")
print(f"K_fold: {K_fold}, test_size: {test_size}, normalize: {normalize}, duplicate: {duplicate}, noise_type: {noise_type}, noise_SNR: {noise_SNR}, random_state: {random_state}")
# create a txt file to save the parameters and noise SNR
with open(os.path.join(save_path, folder_name, 'parameters.txt'), 'w') as f:
    f.write(
        f"K_fold: {K_fold}, test_size: {test_size}, normalize: {normalize}, duplicate: {duplicate}, noise_type: {noise_type}, noise_SNR: {noise_SNR}, random_state: {random_state}")

for i_fold in range(K_fold):
    # initialize dataset handler
    iris_dataset_handler = IrisDatasetHandler(test_size=test_size, normalize=normalize, duplicate=duplicate,
                                                noise_type=noise_type, noise_SNR=noise_SNR, random_state=random_state_list[i_fold])

    # preprocess data
    iris_dataset_handler.preprocess_data()

    # get data
    train_X, train_y, test_X, test_y = iris_dataset_handler.get_preprocessed_data(get_clean=False)
    _, _, test_X_clean, test_y_clean = iris_dataset_handler.get_preprocessed_data(get_clean=True)

    # save data as csv files, save labels as another csv file.
    # create a folder with the name of [i_fold] in the save_path
    save_path_to_fold = os.path.join(save_path, folder_name) + '/' + str(i_fold)
    if not os.path.exists(save_path_to_fold):
        os.makedirs(save_path_to_fold)

    # save data with the name of [i_fold]_train_X.csv, [i_fold]_train_y.csv, [i_fold]_test_X.csv, [i_fold]_test_y.csv
    np.savetxt(os.path.join(save_path_to_fold, 'train_X.csv'), train_X, delimiter=',')
    np.savetxt(os.path.join(save_path_to_fold, 'train_y.csv'), train_y, delimiter=',')
    np.savetxt(os.path.join(save_path_to_fold, 'test_X_noise.csv'), test_X, delimiter=',')
    np.savetxt(os.path.join(save_path_to_fold, 'test_y_noise.csv'), test_y, delimiter=',')
    np.savetxt(os.path.join(save_path_to_fold, 'test_X_clean.csv'), test_X_clean, delimiter=',')
    np.savetxt(os.path.join(save_path_to_fold, 'test_y_clean.csv'), test_y_clean, delimiter=',')

################################################################################







