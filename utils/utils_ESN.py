"""
This project is by Heng Zhang, Kyushu University, Japan.
"""


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time



# define a function to run ESN forward pass
def forword_ESN(dataset, model_ESN, activation_fcn=torch.nn.Identity, T=60, batch_size=1, shuffle=False, device='cpu'):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    W_pop_raw = torch.detach_copy(model_ESN.W_pop) # boardcast the W_pop to the batch size
    W_in_raw = torch.detach_copy(model_ESN.W_in)  # boardcast the W_in to the batch size

    # Iterate through the DataLoader
    output_dynamic_X = []
    output_dynamic_Y = []

    # print("Running ESN forward pass...")
    time.sleep(0.1)
    # for batch_data, batch_labels in tqdm(data_loader):
    for batch_data, batch_labels in data_loader:

        u_in = batch_data.to(device)
        u_in = u_in[:, :, None]
        current_batch_size = batch_data.shape[0]
        W_pop = W_pop_raw.expand(current_batch_size, -1, -1)
        W_in = W_in_raw.expand(current_batch_size, -1, -1)
        # initialize neuron_response
        x = torch.zeros(current_batch_size, model_ESN.population_size, 1, device=device)
        for i in range(T):
            x_update = torch.bmm(W_in, u_in) + torch.bmm(W_pop, x)
            x_update = activation_fcn(x_update)
            x = (1 - model_ESN.leaking_rate) * x + model_ESN.leaking_rate * x_update
        output_dynamic_X.append(x.detach().cpu())
        output_dynamic_Y.append(batch_labels)
    # stack the list
    output_dynamic_X = torch.squeeze(torch.cat(output_dynamic_X))
    output_dynamic_Y = torch.cat(output_dynamic_Y)

    return output_dynamic_X, output_dynamic_Y
