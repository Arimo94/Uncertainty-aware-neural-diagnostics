import torch
import numpy as np

# 3. Negative Log-Likelihood (NLL) loss
def nll_loss_LSTM(y_true, mean, std):
    loss = torch.mean(0.5 * torch.log(2 * np.pi * (std)**2) + 0.5 * ((y_true - mean)**2) / (std)**2) + 5
    return loss
