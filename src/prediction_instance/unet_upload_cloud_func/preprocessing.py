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

#This module holds the functions for downloading imagery, reprojecting, stacking, chunking it for input into firenet.
#It also holds the functions that receive Firenet output which is in the form of 1x64x64 arrays, mosaics, and adds spatial 
#metadata back to it.

def select_blobs(bucket_name='gcp-public-data-goes-16'):
    """
    Selects the appropriate list of blobs from GCP fs, most recent hour's worth of data from GOES MCMIPC bucket.
    Returns: List of selected blobs.
    """
    # Get the current time
    attime = datetime.now()

    # Set up Google Cloud Storage client
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Create a range of directories to check. The GOES bucket is
    # organized by hour of day.
    selected_blobs = []
    for i in range(2):  # Get blobs from current hour and previous hour
        current_time = attime - timedelta(hours=i)
        prefix = f'ABI-L2-MCMIPC/{current_time.year}/{current_time.timetuple().tm_yday:03d}/{current_time.hour:02d}/'
        blobs = bucket.list_blobs(prefix=prefix)
        selected_blobs.extend([blob.name for blob in blobs])

    # Sort the blobs by their timestamp in descending order
    selected_blobs.sort(key=lambda name: name.split('_')[3][1:], reverse=True)

    # Check if there are at least 12 blobs
    if len(selected_blobs) < 12:
        raise Exception(f"Only {len(selected_blobs)} blobs found")

    return selected_blobs[:12]


def create_fs():
    """
    Creates a file system object for GCP. 
    Returns: File system object. fs can be interacted with as though it were a local file system.
    """
    fs = fsspec.filesystem('gcs')
    return fs


def download_blob(fs, bucket_name, blob):
    """
    Download a single blob and load it into an xarray Dataset.
    """
    with fs.open(f'{bucket_name}/{blob}') as f:
        ds = xr.open_dataset(f).load()
        # Apply any necessary preprocessing here
    return ds

def create_median_image_parallel(blob_list, fs, bucket_name='gcp-public-data-goes-16'):
    datasets = []
    # Use ThreadPoolExecutor to parallelize the download and loading of datasets
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_blob = {executor.submit(download_blob, fs, bucket_name, blob): blob for blob in blob_list[10::]}
        for future in concurrent.futures.as_completed(future_to_blob):
            blob = future_to_blob[future]
            try:
                ds = future.result()
                # Apply any region selection or additional preprocessing here if needed
                datasets.append(ds)
            except Exception as exc:
                print(f'{blob} generated an exception: {exc}')
    # Continue with concatenation and median calculation
    concated = xr.concat(datasets, dim='time')
    median_ds = concated.median(dim='time', keep_attrs=True)
    return median_ds

def download_landfire_layers(fs, bucket_name='firenet_reference', blob_name='combined_landfire.nc'):
    """
    Downloads the preprocessed landfire layers. These have been reproject_matched to a GOES CONUS 'template' image, 
    which has itself been reprojected to epsg 5070. Properly loading and accessing the spatial metadata uses a
    trick, openning with xarray, saving to nc, then opening the nc tempfile with rioxarray. This is not a "good"
    approach but for whatever reason the spatial metadata couldn't be accessed otherwise. Trying to open directly with
    rioxarray runs into some interfacing problem with google buckets.
    Returns: Preprocessed landfire layers.
    """

    # Open the blob as a full dataset and load it into memory
    f = fs.open(f'{bucket_name}/{blob_name}')
    ds = xr.open_dataset(f).load()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        # Save the dataset to the temporary file
        ds.to_netcdf(tmpfile.name)

        # Open the temporary file with rioxarray
        landfire_layers = rioxarray.open_rasterio(tmpfile.name)

    return landfire_layers

def reproject_dataset(dataset, landfire_layers):
    """
    Reprojects the dataset to the static layers.
    Note that the technique used here is again creating a tempfile and then opening it with rioxarray.
    This is not a "good" approach but for whatever reason the spatial metadata couldn't be accessed otherwise.
    Trying to open directly with rioxarray runs into some interfacing problem with google buckets.
    Returns: Reprojected dataset.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        # Save the dataset to the temporary file
        dataset.to_netcdf(tmpfile.name)

        # Open the temporary file with rioxarray
        ds_rio = rioxarray.open_rasterio(tmpfile.name)
        
        # Reproject the dataset to the template dataset with landfire layers, landfire layers was generated by
        #  `reprojectmatch_and_stack_landfire_for_bucket.ipynb`
        reprojected_dataset = ds_rio.rio.reproject_match(landfire_layers)

    return reprojected_dataset

def engineer_features(dataset):
    """
    Feature engineers the median dataset, adding more informative bands that are ratios of the spectral channels.
    Returns: Feature engineered dataset.
    """

    vars_to_remove = [f for f in dataset.keys() if f.startswith('DQF_')]
    dataset = dataset.drop_vars(vars_to_remove)
    # Ensure the CRS is preserved by extracting it from the original dataset
    original_crs = dataset.rio.crs

    # Compute the new features
    feat1 = dataset['CMI_C06'] / dataset['CMI_C05']
    feat2 = dataset['CMI_C07'] / dataset['CMI_C05']
    feat3 = dataset['CMI_C07'] / dataset['CMI_C06']
    feat4 = dataset['CMI_C14'] / dataset['CMI_C07']

    # Create a dictionary of the new features
    data_dict = {'feat_6_5': feat1, 'feat_7_5': feat2, 'feat_7_6': feat3, 'feat_14_7': feat4}

    # Add the new features to the dataset
    engineered_dataset = dataset.assign(data_dict)

    # Write the CRS of original_dataset to engineered_dataset, as a global attribute
    engineered_dataset.rio.write_crs(original_crs, inplace=True)

    # Write the CRS to every variable in engineered_dataset, making all var attrs match
    for var in engineered_dataset.data_vars:
        engineered_dataset[var].rio.write_crs(original_crs, inplace=True)

    return engineered_dataset

def stack_datasets(goes_ds, landfire_layers):
    """
    Stacks the GOES ds with the preprocessed landfire layers into a dataset.
    Sets 'grid_mapping' to 'spatial_ref' in the encoding for every data variable in the process.
    Returns: Stacked dataset.
    """
    # Merge the two datasets
    stacked_dataset = xr.merge([goes_ds, landfire_layers])

    # Set 'grid_mapping' to 'spatial_ref' in the encoding dictionary for every data variable
    # Pointing this encoding key to the global spatial_ref value is vital to ensure spatial metadata
    # Gets recognized by future rioxarray and other operations
    for var in stacked_dataset.data_vars:
        stacked_dataset[var].encoding['grid_mapping'] = 'spatial_ref'

    # Optionally, delete 'goes_imager_projection' if it's no longer needed
    if 'goes_imager_projection' in stacked_dataset:
        del stacked_dataset['goes_imager_projection']

    return stacked_dataset

def prune_dataset(ds):
    """
    Drops specified data variables from the dataset.
    #TODO make sure the indices to drop match perfectly the original training

    Parameters:
    - ds: xarray.Dataset to be pruned.
    - vars_to_drop: List of strings representing the names of the variables to drop.

    Returns:
    - ds: xarray.Dataset after dropping the specified variables.
    """
    # Drop the specified variables from the dataset
    var_names_to_drop = [list(ds.data_vars)[i] for i in [0, 11, 16, 17, 18]]

    ds = ds.drop_vars(var_names_to_drop)
    
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
    
    # Create a 0-filled template of the first variable
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
