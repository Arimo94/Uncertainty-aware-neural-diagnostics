import subprocess
import os
import random
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the scripts to run
scripts = {
    "train": "main.py",
    "eval": "Evaluate.py"
}

# Define the settings for each training process
settings_list = [
    {"r": "r0", "M": 1, "epochs_MSE": 20, "epochs_NLL": 20, "initial_seq_length": 100, "max_seq_length": 800, "seq_length_step": 100},
]

# Function to run a script with subprocess
def run_script(action, settings):
    if action not in scripts:
        raise ValueError(f"Invalid action: {action}. Valid actions are {list(scripts.keys())}.")

    # Construct the command
    script_path = scripts[action]
    command = [
        "python", "-Xfrozen_modules=off", script_path,
        "--r", settings["r"],
        "--M", str(settings["M"]),
        "--epochs_MSE", str(settings["epochs_MSE"]),
        "--epochs_NLL", str(settings["epochs_NLL"]),
        "--initial_seq_length", str(settings["initial_seq_length"]),
        "--max_seq_length", str(settings["max_seq_length"]),
        "--seq_length_step", str(settings["seq_length_step"]),
    ]

    print(f"Running: {' '.join(command)}")
    
    # Run the script
    result = subprocess.run(command, text=True)
    if result.returncode == 0:
        print(f"Completed: {' '.join(command)}")
    else:
        print(f"Failed: {' '.join(command)}\nError: {result.stderr}")

if __name__ == "__main__":
    # Run each script sequentially
    for settings in settings_list:
        seed = 41  # Your chosen seed value
        set_random_seed(seed)
        run_script("train", settings)
        run_script("eval", settings)