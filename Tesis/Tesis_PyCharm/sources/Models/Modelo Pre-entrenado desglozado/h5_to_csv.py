import numpy as np
import h5py
import os

# Load the HDF5 model file
model_file = h5py.File('keras_model.h5', 'r')

# Access the 'model_weights' group
model_weights_group = model_file['model_weights']

# Create a dictionary to store the weight values
weights_dict = {}

# Recursively traverse the group structure
def extract_weights(group, prefix=''):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            weights_dict[prefix + key] = item[()]
        elif isinstance(item, h5py.Group):
            extract_weights(item, prefix + key + '/')

# Extract the weights from the hierarchical structure
extract_weights(model_weights_group)
'''
# Create directories if they don't exist
for key, _ in weights_dict.items():
    directory = os.path.dirname(key + '.csv')
    if not os.path.exists(directory):
        os.makedirs(directory)
'''
# Save the weights to separate CSV files
for key, value in weights_dict.items():
    print(f"Key: {key}")
    print(f"Value shape: {value.shape}")
    print(f"Value:\n{value}")
    flattened_value = value.reshape((-1, value.shape[-1]))
    print(f"Flattened value shape: {flattened_value.shape}")
    print(f"Flattened value:\n{flattened_value}")
    np.savetxt(key + '.csv', flattened_value, delimiter=',')
