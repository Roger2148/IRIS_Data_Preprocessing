# IRIS_Data_Preprocessing
## Getting Started
### Tested Environment
```
* Python == 3.8.3
* Scikit-learn == 1.3.0
* Numpy == 1.22.3
* Torch == 1.12.1
* Scipy == 1.7.3
```

### Iris Dataset Gerenation
To generate the Iris dataset, run "iris_load.py".
Set the following parameters to save with a timestamp in the file name.
```
"save_with_timestamp = True"
```
Key parameters are as follows.
```
K_fold = 30         # 30-fold cross validation is used by default.
test_size = 0.5     # The ratio of test data to all data
normalize = False   # If True, the data is normalized
duplicate = 3       # After splitting the data into train and test, the data is duplicated
noise_type = "test" # "test" or "train" or "all"
noise_SNR = 10      # noise level in dB
```

### Classification using Echo State Network
To train the ESN, run "classification_ESN.py".
Check the dataset_name before running. By default, the dataset_name is "backup".

Key parameters are as follows.
```
ESN_input_size = train_X.shape[1] # The number of input neurons (should be 4 in IRIS dataset).
ESN_output_size = 1             
ESN_n_neuron = 128            # The number of reservoir neurons
ESN_sparsity = 0.8
ESN_spectral_radius = 1.1
ESN_leaking_rate = 0.8
ESN_input_scaling = 1.0
ESN_batch_size = 20
ESN_T = 60       # The total time steps of a sample goes through the reservoir 
                 #(60 steps are enough to get a stable reservoir state)
reg = 1e-1  # regularization coefficient
```

### Problem of Data Normalization
In many cases, input data are normalized before being fed into machine learning models.
However, in the case of ESN, normalization is not ideal.
Preliminary experiments show that the performance of ESN is better when the input data are not normalized.
One possible reason is that the input data, 
which is normalized to have zero mean and unit variance, 
falls into a small range for tanh() activation function.
Therefore, the model becomes less non-linear.
By inputting non-normalized data, however, the input data falls into a wider range,
which suits the tanh() activation function better.