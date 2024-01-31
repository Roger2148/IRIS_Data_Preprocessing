import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class IrisDatasetHandler:
    def __init__(self, test_size=0.3, normalize=True, duplicate=1, noise_type="test",
                 noise_SNR=None, random_state=None):

        print(f"[IRIS PREPROCESSING] Initializing IrisDatasetHandler...")

        # set attributes
        self.test_size = test_size
        self.normalize = normalize
        self.duplicate = duplicate
        self.noise = noise_type # noise_type can be "test" or "train", or "all"
        self.noise_SNR = noise_SNR # noise_SNR can be None or a dB.
        self.rng = self.set_random_state(random_state)

        # status
        self.status_preprocessed = False


    def preprocess_data(self, random_state=None):
        #
        print(f"[IRIS PREPROCESSING] Preprocessing data...")

        # load data
        self.data, self.labels = self.load_iris_data() # data and labels will be processed throughout the preprocessing steps
        self.data_raw, self.labels_raw = self.copy_data(data_tuple=(self.data, self.labels)) # make a deep copy

        # shuffle data
        self.data, self.labels = self.shuffle_data(data_tuple=(self.data, self.labels))
        self.data_shuffled, self.labels_shuffled = self.copy_data(data_tuple=(self.data, self.labels)) # make a deep copy

        # split data
        self.data_train, self.data_test, self.labels_train, self.labels_test = self.split_data(data_tuple=(self.data, self.labels))
        self.data_train_raw, self.labels_train_raw = self.copy_data(data_tuple=(self.data_train, self.labels_train)) # make a deep copy
        self.data_test_raw, self.labels_test_raw = self.copy_data(data_tuple=(self.data_test, self.labels_test)) # make a deep copy

        # normalize data
        # the reason why we normalize the data after splitting is that we want to make sure the test data is not used in the normalization process
        if self.normalize:
            self.data_train, self.labels_train = self.normalize_data(data_tuple=(self.data_train, self.labels_train))
            self.data_test, self.labels_test = self.normalize_data(data_tuple=(self.data_test, self.labels_test))
            self.data_train_norm, self.labels_train_norm = self.copy_data(data_tuple=(self.data_train, self.labels_train)) # make a deep copy
            self.data_test_norm, self.labels_test_norm = self.copy_data(data_tuple=(self.data_test, self.labels_test)) # make a deep copy

        # duplicate data
        self.data_train, self.labels_train = self.duplicate_data(data_tuple=(self.data_train, self.labels_train), k=self.duplicate)
        self.data_test, self.labels_test = self.duplicate_data(data_tuple=(self.data_test, self.labels_test), k=self.duplicate)
        self.data_train_dup, self.labels_train_dup = self.copy_data(data_tuple=(self.data_train, self.labels_train)) # make a deep copy
        self.data_test_dup, self.labels_test_dup = self.copy_data(data_tuple=(self.data_test, self.labels_test)) # make a deep copy

        # clean data
        self.data_train_clean, self.labels_train_clean = self.copy_data(data_tuple=(self.data_train_dup, self.labels_train_dup))
        self.data_test_clean, self.labels_test_clean = self.copy_data(data_tuple=(self.data_test_dup, self.labels_test_dup))

        # add noise
        # check if the self.noise is correctly initialized. check if self.noise_SNR is correctly initialized, it should be None or a dB
        if self.noise not in ["train", "test", "all"]:
            warnings.warn(f"[IRIS PREPROCESSING] self.noise is not correctly initialized, it is set to {self.noise}, which is not in ['train', 'test', 'all']. No noise is added...")

        if self.noise == "train":
            self.data_train, self.labels_train = self.add_noise(data_tuple=(self.data_train, self.labels_train), SNR=self.noise_SNR)
            self.data_train_noise, self.labels_train_noise = self.copy_data(data_tuple=(self.data_train, self.labels_train))
        elif self.noise == "test":
            self.data_test, self.labels_test = self.add_noise(data_tuple=(self.data_test, self.labels_test), SNR=self.noise_SNR)
            self.data_test_noise, self.labels_test_noise = self.copy_data(data_tuple=(self.data_test, self.labels_test))
        elif self.noise == "all":
            self.data_train, self.labels_train = self.add_noise(data_tuple=(self.data_train, self.labels_train), SNR=self.noise_SNR)
            self.data_test, self.labels_test = self.add_noise(data_tuple=(self.data_test, self.labels_test), SNR=self.noise_SNR)
            self.data_train_noise, self.labels_train_noise = self.copy_data(data_tuple=(self.data_train, self.labels_train))
            self.data_test_noise, self.labels_test_noise = self.copy_data(data_tuple=(self.data_test, self.labels_test))
        else:
            pass

        # preprocess complete
        self.data_train_preprocessed = self.data_train
        self.labels_train_preprocessed = self.labels_train
        self.data_test_preprocessed = self.data_test
        self.labels_test_preprocessed = self.labels_test

        self.status_preprocessed = True
        print(f"[IRIS PREPROCESSING] Preprocessing complete.")


    def get_preprocessed_data(self, get_clean=False):
        if not self.status_preprocessed:
            raise ValueError("[IRIS PREPROCESSING] Data has not been preprocessed yet. Please call preprocess_data() first.")
        elif get_clean:
            return self.data_train_clean, self.labels_train_clean, self.data_test_clean, self.labels_test_clean
        else:
            return self.data_train_preprocessed, self.labels_train_preprocessed, self.data_test_preprocessed, self.labels_test_preprocessed


    def load_iris_data(self):
        # load data
        data = load_iris()
        return data['data'], data['target']

    def shuffle_data(self, data_tuple):
        # check if data is formatted correctly (data, labels)
        if len(data_tuple) != 2:
            raise ValueError('[IRIS PREPROCESSING] Data must be formatted as (data, labels)')
        # check data is in numpy array format
        if not isinstance(data_tuple[0], np.ndarray) or not isinstance(data_tuple[1], np.ndarray):
            raise ValueError('[IRIS PREPROCESSING] Data must be in numpy array format')

        # unpack data
        data, labels = data_tuple
        # shuffle data
        data, labels = shuffle(data, labels, random_state=self.rng)

        return data, labels

    def split_data(self, data_tuple):
        # check if data is formatted correctly (data, labels)
        if len(data_tuple) != 2:
            raise ValueError('[IRIS PREPROCESSING] Data must be formatted as (data, labels)')
        # check data is in numpy array format
        if not isinstance(data_tuple[0], np.ndarray) or not isinstance(data_tuple[1], np.ndarray):
            raise ValueError('[IRIS PREPROCESSING] Data must be in numpy array format')

        # unpack data
        data, labels = data_tuple
        # split data
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=self.test_size, random_state=self.rng,
                                                                            shuffle=False, stratify=None) # we manually shuffle the data before splitting, so we set shuffle=False.

        return data_train, data_test, labels_train, labels_test

    def copy_data(self, data_tuple):
        # check if data is formatted correctly (data, labels)
        if len(data_tuple) != 2:
            raise ValueError('[IRIS PREPROCESSING] Data must be formatted as (data, labels)')
        # check data is in numpy array format
        if not isinstance(data_tuple[0], np.ndarray) or not isinstance(data_tuple[1], np.ndarray):
            raise ValueError('[IRIS PREPROCESSING] Data must be in numpy array format')

        # unpack data
        data, labels = data_tuple
        # make a deep copy of the data
        data_copy = np.copy(data)
        labels_copy = np.copy(labels)

        return data_copy, labels_copy

    def normalize_data(self, data_tuple):
        # check if data is formatted correctly (data, labels)
        if len(data_tuple) != 2:
            raise ValueError('[IRIS PREPROCESSING] Data must be formatted as (data, labels)')
        # check data is in numpy array format
        if not isinstance(data_tuple[0], np.ndarray) or not isinstance(data_tuple[1], np.ndarray):
            raise ValueError('[IRIS PREPROCESSING] Data must be in numpy array format')

        # unpack data
        data, labels = data_tuple
        # normalize data
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        return data, labels

    def duplicate_data(self, data_tuple, k):
        # check if data is formatted correctly (data, labels)
        if len(data_tuple) != 2:
            raise ValueError('[IRIS PREPROCESSING] Data must be formatted as (data, labels)')
        # check data is in numpy array format
        if not isinstance(data_tuple[0], np.ndarray) or not isinstance(data_tuple[1], np.ndarray):
            raise ValueError('[IRIS PREPROCESSING] Data must be in numpy array format')

        # unpack data
        data, labels = data_tuple
        # duplicate data
        data = np.tile(data, (k, 1))
        labels = np.tile(labels, k)

        return data, labels

    def add_noise(self, data_tuple, SNR):
        # check if data is formatted correctly (data, labels)
        if len(data_tuple) != 2:
            raise ValueError('[IRIS PREPROCESSING] Data must be formatted as (data, labels)')
        # check data is in numpy array format
        if not isinstance(data_tuple[0], np.ndarray) or not isinstance(data_tuple[1], np.ndarray):
            raise ValueError('[IRIS PREPROCESSING] Data must be in numpy array format')
        # check if self.noise_SNR is correctly initialized, it should be None or a dB
        if SNR is not None and not isinstance(SNR, int):
            # warning
            warnings.warn(
                f"[IRIS PREPROCESSING] self.noise_SNR is not correctly initialized, it is set to {SNR}, which is not an integer. No noise is added...")

        # unpack data
        data, labels = data_tuple
        # add noise
        if SNR is None:
            print(f"[IRIS PREPROCESSING] SNR is None, no noise is added...")
        else:
            print(f"[IRIS PREPROCESSING] SNR is {SNR} dB, noise is added...")
            # calculate noise power
            signal_power = np.mean(data ** 2)
            noise_power = signal_power / (10 ** (SNR / 10))
            # generate noise
            if self.rng is None:
                noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
            else:
                noise = self.rng.normal(0, np.sqrt(noise_power), data.shape)
            # add noise
            data = data + noise

        return data, labels


    def set_random_state(self, random_state):
        self.rng = None if random_state is None else np.random.RandomState(random_state)
        if self.rng is None:
            print(f"[IRIS PREPROCESSING] Random state is set to None...")
        else:
            print(f"[IRIS PREPROCESSING] Random state has been set to {random_state} --> {self.rng}...")

        return self.rng


