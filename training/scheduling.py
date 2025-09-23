import torch
from utils.data_utils import create_sequences
from torch.utils.data import DataLoader, TensorDataset

def dynamic_seq_length_training(train_function, models, optimizers, combined_data, Causal, r, initial_seq_length, max_seq_length, seq_length_step, n_epochs, batch_size):
    num_steps = (max_seq_length - initial_seq_length) // seq_length_step
    epochs_per_step = n_epochs // (num_steps + 1)  # +1 to include the initial phase
    current_seq_length = initial_seq_length
    for step in range(num_steps + 1):  # +1 to include the initial training phase
        epoch_start = step * epochs_per_step
        epoch_end = min((step + 1) * epochs_per_step, n_epochs)
        
        # Adjust sequence length
        if step > 0:  # Increase seq_length after the first step
            current_seq_length = min(current_seq_length + seq_length_step, max_seq_length)
            print(f"\nIncreasing seq_length to {current_seq_length} for epochs {epoch_start+1} to {epoch_end}")
        
        # Recreate sequences and DataLoader
        X, y = create_sequences(
            combined_data['wltp_NF'][:, Causal[r][0:-1]],
            combined_data['wltp_NF'][:, Causal[r][-1]],
            current_seq_length
        )
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train ensemble for the specified range of epochs
        train_function(models, optimizers, dataloader, epoch_end - epoch_start)
        

def fixed_seq_length_training(train_function, models, optimizers, combined_data, Causal, r, seq_length, n_epochs, batch_size):
    current_seq_length = seq_length   
    # Recreate sequences and DataLoader
    X, y = create_sequences(
        combined_data['wltp_NF'][:, Causal[r][0:-1]],
        combined_data['wltp_NF'][:, Causal[r][-1]],
        current_seq_length
    )
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train ensemble for the specified range of epochs
    train_function(models, optimizers, dataloader, n_epochs)