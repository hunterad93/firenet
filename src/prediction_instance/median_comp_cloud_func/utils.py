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
    # gcs_options = {'token': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}
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
                ds = ds.isel(x=slice(0, 1250), y=slice(0, 1250))
                datasets.append(ds)
            except Exception as exc:
                print(f'{blob} generated an exception: {exc}')
    # Continue with concatenation and median calculation
    concated = xr.concat(datasets, dim='time')
    median_ds = concated.median(dim='time', keep_attrs=True)
    print('median_ds_created')
    # Slice the median dataset for the specified x and y dimensions
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

def save_median_ds_to_bucket(median_ds, fs, bucket_name, blob_name):
    """
    Saves the median dataset to a specified GCP bucket.
    
    Args:
    median_ds (xarray.Dataset): The median dataset to be saved.
    fs (fsspec filesystem object): Filesystem object to interact with GCP.
    bucket_name (str): The name of the bucket where the dataset will be saved.
    blob_name (str): The name of the blob (file) in the bucket.
    """
    # Convert the xarray Dataset to a temporary netCDF file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmpfile:
        median_ds.to_netcdf(tmpfile.name)
        # Ensure the file is closed before uploading
        tmpfile.close()
        
        # Upload the file to the bucket
        with fs.open(f'{bucket_name}/{blob_name}', 'wb') as f:
            with open(tmpfile.name, 'rb') as tmpfile_read:
                f.write(tmpfile_read.read())
                
        # Clean up the temporary file
        os.remove(tmpfile.name)
