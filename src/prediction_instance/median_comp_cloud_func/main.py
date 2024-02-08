import utils
import gc

def median_composite(request):
    print('preprocessing')
    # Selects the most recent hour's worth of satellite imagery blobs from GCP
    selected_blobs = utils.select_blobs()
    # Creates a file system object for interacting with GCP as if it were a local file system
    fs = utils.create_fs()
    # Downloads and processes the selected blobs in parallel to create a median image dataset
    median_ds = utils.create_median_image_parallel(selected_blobs, fs)
    # Downloads preprocessed landfire layers
    landfire_layers = utils.download_landfire_layers(fs)
    # Reprojects the median dataset to match the spatial reference of the landfire layers
    reprojected_median_ds = utils.reproject_dataset(median_ds, landfire_layers)
    del median_ds  # Free up memory
    gc.collect()
    # Adds additional features to the dataset by calculating ratios of spectral channels
    reprojected_median_ds = utils.engineer_features(reprojected_median_ds)
    # Merges the GOES dataset with the landfire layers into a single dataset
    stacked_ds = utils.stack_datasets(reprojected_median_ds, landfire_layers)
    del reprojected_median_ds  # Free up memory
    del landfire_layers  # Free up memory
    gc.collect()
    # Removes unnecessary variables from the dataset to match the original training data structure
    stacked_ds = utils.prune_dataset(stacked_ds)
    # Downloads preprocessed landfire layers
    utils.save_median_ds_to_bucket(stacked_ds, fs, "firenet_reference", "stacked_ds.nc")
    return 'Function executed successfully', 200