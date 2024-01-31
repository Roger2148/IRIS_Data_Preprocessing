"""
This project is by Heng Zhang, Kyushu University, Japan.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import pairwise_distances, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from models.reservoir.model_ESN_torch import ESN
from utils.utils_ESN import forword_ESN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load data
dataset_name = 'backup'
# dataset_name = '20240131122318'  # uncomment this line to load a specific dataset
dir_iris = os.path.join(os.getcwd(), 'data', 'iris_data', dataset_name)

# read the "parameters.txt" file
with open(os.path.join(dir_iris, 'parameters.txt'), 'r') as f:
    parameters = f.readlines()
# print out parameters
print("===================================================================")
print("[IRIS PREPROCESSING] Parameters:")
print(parameters)

K_fold = 30
# calculate in folder dir_iris, how many sub-folders are there (i.e., ignore the files and only count the folders)
dir_list_all = os.listdir(dir_iris)
dir_list = []
for dir in dir_list_all:
    if os.path.isdir(os.path.join(dir_iris, dir)):
        dir_list.append(dir)
# sort the dir_list
dir_list.sort()

# check if there are more than K_fold sub-folders in the dir_iris. If so, use the first K_fold sub-folders in the list, else, raise error
if len(dir_list) < K_fold:
    raise ValueError(f"Number of folders in {dir_iris} is less than K_fold, please check the data.")
else:
    folder_list = dir_list[:K_fold]

# config parameters
onehot_encoder = OneHotEncoder(sparse_output=False)
acc_list_train = []
acc_list_test_clean = []
acc_list_test_noise = []

for i_fold in range(K_fold):
    print("===================================================================")
    print(f"Running fold {i_fold}...")

    # load data from current folder. The data is saved as several .csv files.
    dir_current = os.path.join(dir_iris, folder_list[i_fold])
    # load data
    train_X, train_y = np.loadtxt(os.path.join(dir_current, 'train_X.csv'), delimiter=','), np.loadtxt(os.path.join(dir_current, 'train_y.csv'), delimiter=',')
    test_X_noise, test_y_noise = np.loadtxt(os.path.join(dir_current, 'test_X_noise.csv'), delimiter=','), np.loadtxt(os.path.join(dir_current, 'test_y_noise.csv'), delimiter=',')
    test_X_clean, test_y_clean = np.loadtxt(os.path.join(dir_current, 'test_X_clean.csv'), delimiter=','), np.loadtxt(os.path.join(dir_current, 'test_y_clean.csv'), delimiter=',')

    # # linear transformation
    # amplifier = 1
    # bias = 3
    # train_X = train_X * amplifier + bias
    # test_X_noise = test_X_noise * amplifier + bias
    # test_X_clean = test_X_clean * amplifier + bias

    # construct dataset
    train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    test_dataset_noise = TensorDataset(torch.tensor(test_X_noise, dtype=torch.float32), torch.tensor(test_y_noise, dtype=torch.long))
    test_dataset_clean = TensorDataset(torch.tensor(test_X_clean, dtype=torch.float32), torch.tensor(test_y_clean, dtype=torch.long))

    # initialize ESN
    ESN_input_size = train_X.shape[1]
    ESN_output_size = 1
    ESN_n_neuron = 128
    ESN_sparsity = 0.8
    ESN_spectral_radius = 1.1
    ESN_leaking_rate = 0.8
    ESN_input_scaling = 1.0
    ESN_T = 60
    ESN_batch_size = 20

    model_ESN = ESN(input_size=ESN_input_size, output_size=ESN_output_size, population_size=ESN_n_neuron,
                         sparsity=ESN_sparsity, spectral_radius=ESN_spectral_radius,
                         leaking_rate=torch.tensor([ESN_leaking_rate], device=device),
                         input_scaling=ESN_input_scaling)

    # forward pass
    print("Running ESN forward pass...")
    time.sleep(0.1)
    train_ESN_X, train_ESN_Y = forword_ESN(train_dataset, model_ESN, activation_fcn=torch.tanh,
                                             T=ESN_T, batch_size=ESN_batch_size, shuffle=False, device=device)

    # get one-hot encoding of output_ESN_Y
    train_ESN_Y_onehot = torch.tensor(onehot_encoder.fit_transform(train_ESN_Y[:, None]), dtype=torch.float32)
    # Ridge regression
    reg = 1e-1  # regularization coefficient
    W_out = torch.linalg.solve(torch.matmul(train_ESN_X.T, train_ESN_X) + reg * torch.eye(train_ESN_X.shape[1]),
                          torch.matmul(train_ESN_X.T, train_ESN_Y_onehot)).T

    # test the train data (see overfitting)
    pred_train = torch.argmax(torch.matmul(W_out, train_ESN_X.T), dim=0)
    acc_pred = accuracy_score(pred_train, train_ESN_Y)
    print(f"Training accuracy: {acc_pred}")

    # test clean data
    test_ESN_X_clean, test_ESN_Y_clean = forword_ESN(test_dataset_clean, model_ESN, activation_fcn=torch.tanh,
                                                T=ESN_T, batch_size=ESN_batch_size, shuffle=False, device=device)
    # get one-hot encoding of output_ESN_Y
    test_ESN_Y_onehot_clean = torch.tensor(onehot_encoder.fit_transform(test_ESN_Y_clean[:, None]), dtype=torch.float32)
    pred_test_clean = torch.argmax(torch.matmul(W_out, test_ESN_X_clean.T), dim=0)
    acc_pred_clean = accuracy_score(pred_test_clean, test_ESN_Y_clean)
    print(f"Testing accuracy: {acc_pred_clean}")

    # test noisy data
    test_ESN_X_noise, test_ESN_Y_noise = forword_ESN(test_dataset_noise, model_ESN, activation_fcn=torch.tanh,
                                                T=ESN_T, batch_size=ESN_batch_size, shuffle=False, device=device)
    # get one-hot encoding of output_ESN_Y
    test_ESN_Y_onehot_noise = torch.tensor(onehot_encoder.fit_transform(test_ESN_Y_noise[:, None]), dtype=torch.float32)
    pred_test_noise = torch.argmax(torch.matmul(W_out, test_ESN_X_noise.T), dim=0)
    acc_pred_noise = accuracy_score(pred_test_noise, test_ESN_Y_noise)
    print(f"Testing accuracy: {acc_pred_noise}")

    # append
    acc_list_train.append(acc_pred)
    acc_list_test_clean.append(acc_pred_clean)
    acc_list_test_noise.append(acc_pred_noise)


    # tsne
    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=2, random_state=None)
    # train_ESN_X_tsne = tsne.fit_transform(train_ESN_X)
    # plt.figure()
    # plt.scatter(train_ESN_X_tsne[:, 0], train_ESN_X_tsne[:, 1], c=train_ESN_Y)
    # plt.show()
    #
    # test_ESN_X_tsne = tsne.fit_transform(test_ESN_X)
    # plt.figure()
    # plt.scatter(test_ESN_X_tsne[:, 0], test_ESN_X_tsne[:, 1], c=test_ESN_Y)
    # plt.show()

# print average accuracy
acc_avg_train = np.mean(acc_list_train)
std_avg_train = np.std(acc_list_train)
acc_avg_test_clean = np.mean(acc_list_test_clean)
std_avg_test_clean = np.std(acc_list_test_clean)
acc_avg_test_noise = np.mean(acc_list_test_noise)
std_avg_test_noise = np.std(acc_list_test_noise)
print("===================================================================")
print(f"Average training accuracy (with s.t.d): {acc_avg_train:2f} ± {std_avg_train:2f}")
print(f"Average testing clean set accuracy (with s.t.d): {acc_avg_test_clean:2f} ± {std_avg_test_clean:2f}")
print(f"Average testing noise set accuracy (with s.t.d): {acc_avg_test_noise:2f} ± {std_avg_test_noise:2f}")

pause = 0