import os
import torch
from models.probabilistic_rnn import Probabilistic_RNN
from utils.data_utils import load_and_normalize_data
from training.train_ensemble import train_ensemble_NLL, train_ensemble_MSE
from torch.utils.data import DataLoader, TensorDataset
import argparse
import torch.optim as optim
from training.scheduling import dynamic_seq_length_training, fixed_seq_length_training




# Configuration
os.chdir(os.path.dirname(os.path.abspath(__file__)))
directory = 'Engine_data/trainingdata/'

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train Ensemble of Probabilistic RNNs")
parser.add_argument("--r", type=str, required=True, help="The Causal set identifier (e.g., r5)")
parser.add_argument("--M", type=int, required=True, help="Number of models in the ensemble")
parser.add_argument("--epochs_MSE", type=int, required=True, help="Number of training epochs for MSE loss")
parser.add_argument("--epochs_NLL", type=int, required=True, help="Number of training epochs for NLL penalty")
parser.add_argument("--initial_seq_length", type=int, default=100, help="Initial sequence length for training")
parser.add_argument("--max_seq_length", type=int, default=400, help="Maximum sequence length for training")
parser.add_argument("--seq_length_step", type=int, default=100, help="Increment for sequence length")
args = parser.parse_args()

# Assign arguments to variables
r = args.r
M = args.M
n_epochs_MSE = args.epochs_MSE
n_epochs_NLL = args.epochs_NLL
initial_seq_length = args.initial_seq_length
max_seq_length = args.max_seq_length
seq_length_step = args.seq_length_step
batch_size = 128


combined_data = load_and_normalize_data(directory)
Causal = {
    'r0': [2,3,4,5,6,7,8,9,10,1]
}



# Initialize ensemble of models
models = [Probabilistic_RNN(input_dim=len(Causal[r][0:-1])) for _ in range(M)]
optimizers = [optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8) for model in models]


# Call the unified function for each training method
print("Training with MSE Loss:")
dynamic_seq_length_training(
    train_function=train_ensemble_MSE,
    models=models,
    optimizers=optimizers,
    combined_data=combined_data,
    Causal=Causal,
    r=r,
    initial_seq_length=initial_seq_length,
    max_seq_length=max_seq_length,
    seq_length_step=seq_length_step,
    n_epochs=n_epochs_MSE,
    batch_size=batch_size
)

print("Training with NLL Penalty:")
fixed_seq_length_training(
    train_function=train_ensemble_NLL,
    models=models,
    optimizers=optimizers,
    combined_data=combined_data,
    Causal=Causal,
    r=r,
    seq_length=max_seq_length,
    n_epochs=n_epochs_NLL,
    batch_size=batch_size
)



def save_ensemble(models, r, M, n_epochs_MSE, n_epochs_NLL, \
    initial_seq_length, max_seq_length, seq_length_step, base_dir="save_models"):
    # Construct the initial directory name
    base_save_dir = os.path.join(base_dir, f"ensemble_{r}_M{M}_epochs{n_epochs_MSE}_{n_epochs_NLL}_seq{initial_seq_length}_{max_seq_length}_{seq_length_step}")
    save_dir = base_save_dir

    # Create the final unique directory
    os.makedirs(save_dir, exist_ok=True)

    # Save each model
    for idx, model in enumerate(models):
        save_path = os.path.join(save_dir, f"model_{idx}.pth")
        torch.save(model.state_dict(), save_path)  # Save the state dictionary
        print(f"Model {idx} saved to {save_path}")

    print(f"All models saved in directory: {save_dir}")
        # Save the hyperparameters and settings to a text file
    hyperparams_path = os.path.join(save_dir, "training_settings.txt")
    with open(hyperparams_path, "w") as f:
        f.write(f"Training Settings\n")
        f.write(f"=================\n")
        f.write(f"r: {r}\n")
        f.write(f"M: {M}\n")
        f.write(f"n_epochs_MSE: {n_epochs_MSE}\n")
        f.write(f"n_epochs_NLL: {n_epochs_NLL}\n")
        f.write(f"initial_seq_length: {initial_seq_length}\n")
        f.write(f"max_seq_length: {max_seq_length}\n")
        f.write(f"seq_length_step: {seq_length_step}\n")

    print(f"Training settings saved to {hyperparams_path}")

# # Save the trained ensemble
save_ensemble(models, r=r, M=M, n_epochs_MSE=n_epochs_MSE, n_epochs_NLL=n_epochs_NLL,\
              initial_seq_length = initial_seq_length, max_seq_length = max_seq_length, seq_length_step = seq_length_step, base_dir="save_models")

print("\nTraining complete!")

