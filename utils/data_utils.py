import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
import os


def create_sequences(data_x, data_y, seq_length=10):
    xs, ys = [], []
    for i in range(len(data_x) - seq_length):
        xs.append(data_x[i:i+seq_length])
        ys.append(data_y[i:i+seq_length])
    return np.array(xs), np.array(ys)

def load_and_normalize_data(directory):
    # Initialize a dictionary to store all the loaded data
    combined_data = {}

    # Iterate through all .mat files in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.mat'):  # Check if the file is a .mat file
            file_path = os.path.join(directory, filename)
            
            # Load the .mat file
            data = loadmat(file_path)
            
            # Extract the 'Data' variable
            if 'Data' in data:
                loaded_data = data['Data']
                print(f"Loaded data from {filename}.")
                
                # Store the data in the dictionary with the filename as the key (without extension)
                variable_name = os.path.splitext(filename)[0]
                combined_data[variable_name] = loaded_data
            else:
                print(f"'Data' variable not found in {filename}.")
    
    # normalize in respect to maximum of Data_NF
    scales = np.max(combined_data['wltp_NF'], axis=0)
    for data_i in combined_data:
        combined_data[data_i] = combined_data[data_i] / scales

            
    return combined_data



def sorting_key(combined_data):
    desired_order = ['wltp_f_iml_6mm' , 'wltp_f_waf_105', 'wltp_f_waf_110', 'wltp_f_pim_080', 'wltp_f_pim_090', 'wltp_f_pic_090', 'wltp_f_pic_110', 'wltp_NF']
    combined_data_keys = sorted(combined_data.keys())
    stripped_keys = {key.replace('Data_', ''): key for key in combined_data_keys}
    data_name = [stripped_keys[name] if name in stripped_keys else f'Data_{name}' for name in desired_order]
    return data_name
