from google.cloud import storage
from datetime import datetime, timedelta
import pandas as pd
import rioxarray
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import xarray as xr
import numpy as np
import fsspec
import os
import tempfile

#This module holds the functions for downloading the imagery stack, chunking it into nparrays for input into firenet.

def create_fs():
    """
    Creates a file system object for GCP. 
    Returns: File system object. fs can be interacted with as though it were a local file system.
    """
    # gcs_options = {'token': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}
    fs = fsspec.filesystem('gcs')
    return fs


def download_blob(fs, bucket_name='firenet_reference', blob='stacked_ds.nc'):
    """
    Download a single blob and load it into an xarray Dataset. Points to median_ds from cloud func output,
    redirected if creating median in memory in create_median_image_parallel. This allows it to be
    compatable either with a separate function creating median, or creating it in this function. This
    dual-function was developed while struggling with memory trying to do it all in one.
    """

    f = fs.open(f'{bucket_name}/{blob}')
    ds = xr.open_dataset(f).load()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        # Save the dataset to the temporary file
        ds.to_netcdf(tmpfile.name)

        # Open the temporary file with rioxarray
        ds = rioxarray.open_rasterio(tmpfile.name)
    return ds

def create_spatial_template(dataset):
    """
    Creates a spatial template from the original dataset by keeping only one data variable
    and setting its values to 0, while preserving spatial metadata. This step allows us to
    Take the datavars out as a 21x64x64 numpy array, then add the model output back as 1x64x64
    so that the spatial projection of the output is untouched.
    
    Parameters:
    - dataset: xarray.Dataset or rioxarray object with spatial dimensions and CRS information.
    
    Returns:
    - template: xarray.Dataset with a single data variable filled with NaNs and original spatial metadata.
    """
    # Clone the dataset to avoid modifying the original
    template = dataset.copy()
    
    # Select the first data variable (assuming there's at least one)
    first_var_name = list(template.data_vars)[0]
    first_var = template[first_var_name]
    
    # Create a nan-filled template of the first variable
    nan_template = xr.full_like(first_var, fill_value=np.nan)
    
    # Remove all data variables from the template
    for var_name in list(template.data_vars):
        del template[var_name]
    
    # Add the NaN-filled template variable back
    template[first_var_name] = nan_template
    
    # Ensure the spatial metadata is preserved
    # Note: This step might be redundant if the metadata is already attached to the coordinates
    # and not the data variables themselves. However, it's a safeguard for maintaining CRS.
    if hasattr(dataset, 'rio') and hasattr(dataset.rio, 'crs'):
        template.rio.write_crs(dataset.rio.crs, inplace=True)
    
    return template

def extract_data_as_array(dataset):
    """
    This function pulls the data variable value arrays out of the xarray dataset.
    """
    # Stack the data variables, then use np.squeeze() to remove the singleton dimension that was a placeholder for time.
    # The resultant array should be the shape 42, 3506, 2266. The second two dimensions may change if region of interest changes.

    stacked_array = np.stack([dataset[var].values for var in dataset.data_vars], axis=0)
    squeezed_array = np.squeeze(stacked_array)
    return squeezed_array

def chunk_ndarray(arr, chunk_size=64):
    """
    Breaks down an N-dimensional array into chunks along the last two dimensions,
    keeping the first dimension intact in each chunk.
    
    Parameters:
    - arr: Input N-dimensional NumPy array with shape (Variables, Height, Width).
    - chunk_size: Size of the chunks along each of the last two dimensions.
    
    Returns:
    - A list of chunks, where each chunk is an N-dimensional NumPy array with shape (Variables, chunk_size, chunk_size).
    """
    chunks = []
    # Iterate over the last two dimensions in steps of `chunk_size`
    for i in range(0, arr.shape[1], chunk_size):  # Height dimension
        for j in range(0, arr.shape[2], chunk_size):  # Width dimension
            # Calculate the end indices while ensuring they do not exceed the array's dimensions
            end_i = min(i + chunk_size, arr.shape[1])
            end_j = min(j + chunk_size, arr.shape[2])
            # Extract the chunk
            chunk = arr[:, i:end_i, j:end_j]
            # Only add chunks that meet the full size requirement (i.e., 42x64x64)
            if chunk.shape[1] == chunk_size and chunk.shape[2] == chunk_size:
                chunks.append(chunk)
    return chunks
