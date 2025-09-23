from tqdm import tqdm
import torch
from utils.loss_functions import nll_loss_LSTM  # Import the loss function here


def train_ensemble_NLL(models, optimizers ,dataloader, n_epochs, lr=0.01):
    for model, optimizer in zip(models, optimizers):
        model.train()
        for epoch in range(n_epochs):
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    mean_pred, std_pred = model(batch_x)
                    loss = nll_loss_LSTM(batch_y, mean_pred.squeeze(dim=-1), std_pred.squeeze())
                    loss.backward()
                    optimizer.step()
                if epoch % (n_epochs/5) == 0:
                    print(f'Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}')
                    


def train_ensemble_MSE(models, optimizers, dataloader, n_epochs, lr=0.01):
    for model, optimizer in zip(models, optimizers):
        model.train()
        for epoch in range(int(n_epochs)):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                mean_pred, _ = model(batch_x)  # Ignore std_pred
                loss = torch.nn.functional.mse_loss(mean_pred.squeeze(dim=-1), batch_y)  # MSE loss for mean
                loss.backward()
                optimizer.step()
            if epoch % (n_epochs / 5) == 0:
                print(f'Phase 1 - Epoch [{epoch}/{n_epochs}], MSE Loss: {loss.item():.4f}')

