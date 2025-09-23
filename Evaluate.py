import os
import torch
import numpy as np
from utils.data_utils import create_sequences, sorting_key
from models.probabilistic_rnn import Probabilistic_RNN
import pickle  # To save results
import argparse


os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Load ensemble models
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


def load_ensemble(save_dir, r, M):
    ensemble = []
    for idx in range(M):
        model_path = os.path.join(save_dir, f"model_{idx}.pth")
        model = Probabilistic_RNN(input_dim=len(Causal[r][0:-1]))  # Replace with correct input dimension
        model.load_state_dict(torch.load(model_path))
        model.eval()
        ensemble.append(model)
        print(f"Model {idx} loaded from {model_path}")
    return ensemble

# Evaluate ensemble predictions
def evaluate_ensemble(ensemble, combined_data, data_name, Causal, r, save_dir, seq_length=10):
    # Initialize variables to store results for all datasets
    all_mu_star = []
    all_sigma_star = []
    all_U_aleatoric = []
    all_U_epistemic = []
    all_y_test = []
    all_x_test = []
    all_r = []
    all_mu = []
    all_sigma = []
    # Iterate through all datasets in data_name
    for data_i in data_name:
        # Prepare test data for the current dataset
        X_t, y_t = create_sequences(
            combined_data[data_i][:,Causal[r][0:-1]],
            combined_data[data_i][:,Causal[r][-1]],
            seq_length=len(combined_data[data_i][:, Causal[r][-1]]) - 1
        )
        x_test = torch.tensor(X_t, dtype=torch.float32)
        y_test = torch.tensor(y_t, dtype=torch.float32)
        all_y_test.append(y_test[0, :])
        all_x_test.append(x_test[0, :])

        # Collect predictions from the ensemble
        ensemble_means = []
        ensemble_stds = []

        for model in ensemble:
            model.eval()
            with torch.no_grad():
                mean_pred, std_pred = model(x_test)
                ensemble_means.append(mean_pred.numpy())
                ensemble_stds.append(std_pred.numpy())
        
        # Compute ensemble metrics
        mu_star = np.mean(ensemble_means, axis=0)
        sigma_star = np.sqrt(
            np.sum(np.array(ensemble_means)**2 + np.array(ensemble_stds)**2, axis=0) / len(ensemble) - mu_star**2
        )
        U_aleatoric = np.mean(np.array(ensemble_stds)**2, axis=0)
        U_epistemic = np.log(sigma_star**2) - np.mean(np.log(np.array(ensemble_stds)**2), axis=0)
        
        # Store results for the current dataset
        all_mu_star.append(mu_star[0,:,0])
        all_sigma_star.append(sigma_star[0,:,0])
        all_U_aleatoric.append(U_aleatoric[0,:,0])
        all_U_epistemic.append(U_epistemic[0,:,0])
        all_r.append(mu_star[0,:,0] - y_test[0, :].detach().cpu().numpy())
        all_mu.append(ensemble_means)
        all_sigma.append(ensemble_stds)

    # Save results
    results = {
        "all_r": all_r,
        "all_mu_star": all_mu_star,
        "all_sigma_star": all_sigma_star,
        "all_U_aleatoric": all_U_aleatoric,
        "all_U_epistemic": all_U_epistemic,
        "all_y_test": all_y_test,
        "all_x_test": all_x_test,
        "all_mu": all_mu,
        "all_sigma": all_sigma
    }
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"results_{r}_M{len(ensemble)}_epochs{n_epochs_MSE}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}")

# Main function
if __name__ == "__main__":
    
    
    base_dir = "save_models"  # Directory where models are saved
    combined_data_dir = "Engine_data/trainingdata/"  # Directory for data
    save_dir = os.path.join(base_dir, f"ensemble_{r}_M{M}_epochs{n_epochs_MSE}_{n_epochs_NLL}_seq{initial_seq_length}_{max_seq_length}_{seq_length_step}")
    
    # Load data
    from utils.data_utils import load_and_normalize_data
    combined_data = load_and_normalize_data(combined_data_dir)

    # Define causal relationships (replace with your actual causal mapping)
    Causal = {
        'r0': [2,3,4,5,6,7,8,9,10,1]
    }
    
    data_name = sorting_key(combined_data)
    # Load ensemble
    ensemble = load_ensemble(save_dir, r, M)
    # Evaluate ensemble and save results
    evaluate_ensemble(ensemble, combined_data, data_name, Causal, r, save_dir)
