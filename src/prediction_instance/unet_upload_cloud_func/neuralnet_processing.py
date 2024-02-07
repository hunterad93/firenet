import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from xarray.backends import NetCDF4DataStore
from UNET_initiator import Unet
from google.cloud import storage
from io import BytesIO


def load_unet_model(bucket_name='firenet_reference', blob_name='model2.pt'):
    # Set up Google Cloud Storage client
    client = storage.Client()

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download the model as a byte stream
    byte_stream = BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)  # Move to the beginning of the byte stream
    
    # Load the model directly from the byte stream
    unet = Unet(21, 1)
    unet.load_state_dict(torch.load(byte_stream, map_location=torch.device('cpu')))
    return unet

def normalize_bands(np_arr):
    """
    Normalizes each band in a 3D array independently.
    
    Parameters:
    - np_arr: A 3D NumPy array of shape (bands, height, width).
    
    Returns:
    - A 3D NumPy array of the same shape, with each band normalized.
    """
    normalized = np.zeros_like(np_arr, dtype=np.float32)
    for i in range(np_arr.shape[0]):
        band = np_arr[i]
        mini, maxi = np.min(band), np.max(band)
        if maxi != mini:
            normalized[i] = (band - mini) / (maxi - mini)
        else:
            normalized[i] = np.nan
    return normalized

def predict(chunks, unet):
    predictions = []
    # `chunks` is a list of your 3D arrays (21x64x64)
    for chunk in chunks:
        # Normalize the current chunk
        normalized_chunk = normalize_bands(chunk)
        
        # Convert the normalized chunk to a PyTorch tensor
        input_data = torch.from_numpy(normalized_chunk).float()
        
        # Add a batch dimension, just to fit with the model this is a singleton
        input_data = input_data.unsqueeze(0)
        
        
        # Now, input_data is ready to be fed into the U-Net model
        # Append the prediction for the current chunk to a list of predictions
        predictions.append(unet(input_data))
        
    return predictions
