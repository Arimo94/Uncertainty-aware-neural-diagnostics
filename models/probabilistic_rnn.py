import torch.nn as nn
import torch

class Probabilistic_RNN(nn.Module):
    def __init__(self, input_dim):
        super(Probabilistic_RNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers = 1, batch_first=True)
        self.fc_mean = nn.Linear(64, 1)
        self.fc_std = nn.Linear(64, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        mean = self.fc_mean(lstm_out)
        std = torch.log(1 + torch.exp(self.fc_std(lstm_out)))  # Exponentiate std to ensure it's positive
        return mean, std
    