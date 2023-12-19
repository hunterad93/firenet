
import fsspec
import xarray as xr
import numpy as np
import geojson
from google.cloud import storage
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery

def get_lat_lon_mapping(bucket_name, blob_name):
    """
    Fetches a dataset from a specified blob in a Google Cloud Storage bucket and extracts latitude and longitude mappings.
    GOES uses a fixed grid so the mappings are always the same. 
    
    :param bucket_name: The name of the Google Cloud Storage bucket to fetch the blob from
    :param blob_name: The name of the blob to fetch the dataset from
    :return: A tuple of two numpy arrays representing the latitude and longitude mappings respectively
    """
    fs = fsspec.filesystem('gcs')
    with fs.open(f'{bucket_name}/{blob_name}') as f:
        ds = xr.open_dataset(f)
        lat_mapping = ds['latitude'].values.astype(float)
        lon_mapping = ds['longitude'].values.astype(float)
    return lat_mapping, lon_mapping

def get_most_recent_blob_name(bucket_name='gcp-public-data-goes-16'):
    """
    Fetches the name of the most recent blob from a specified Google Cloud Storage bucket.
    Deals with edge cases where the most recent blob may not be in the most recent hour.

    :param bucket_name: The name of the Google Cloud Storage bucket to fetch the blob from. Default is 'gcp-public-data-goes-16'.
    :return: The name of the most recent blob. If no blobs are found, returns None.
    """
    # Set up Google Cloud Storage client
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Get the current date and time
    current_time = datetime.utcnow() + timedelta(hours=5)

    # Create a range of directories to check. The GOES bucket is
    # organized by hour of day.
    blob_names = []
    for hours in range(10):  # Check the last 10 hours starting from 5 in future, just to make sure we don't miss anything
        check_time = current_time - timedelta(hours=hours)
        prefix = f'ABI-L2-FDCC/{check_time.year}/{check_time.timetuple().tm_yday:03d}/{check_time.hour:02d}/'
        blobs = bucket.list_blobs(prefix=prefix)
        blob_names.extend([blob.name for blob in blobs])

    # Sort blob names by timestamp
    blob_names.sort(key=lambda name: name.split('_')[3][1:], reverse=True)  # Extract timestamp after 's'

    # Return the most recent blob name
    return blob_names[0] if blob_names else None

def get_dataset(blob_name, fs, bucket_name='gcp-public-data-goes-16'):
    """
    Opens a specified blob from a Google Cloud Storage bucket as an xarray Dataset.

    :param blob_name: The name of the blob to open
    :param fs: The fsspec filesystem object to use for opening the blob
    :param bucket_name: The name of the Google Cloud Storage bucket to fetch the blob from. Default is 'gcp-public-data-goes-16'.
    :return: An xarray Dataset representing the data in the blob
    """
    # Open the blob as an xarray Dataset
    f = fs.open(f'{bucket_name}/{blob_name}')
    ds = xr.open_dataset(f, engine='h5netcdf')
    return ds


def generate_geojson_points(ds, lat_mapping, lon_mapping):
    """
    Processes a dataset to generate GeoJSON points for each location where the 'DQF' value is 0.
    DQF 0 indicates high confidence that the pixel is a fire.

    :param ds: The xarray Dataset to process
    :param lat_mapping: A numpy array representing the latitude mapping
    :param lon_mapping: A numpy array representing the longitude mapping
    :return: A tuple containing the timestamp from the dataset and a string representing the generated GeoJSON
    """
    # Process the data to generate GeoJSON
    band_data = ds['DQF'].values
    zero_indices = np.where(band_data == 0)
    zero_lat_lon = lat_mapping[zero_indices], lon_mapping[zero_indices]
    features = []
    for lat, lon in zip(*zero_lat_lon):
        point = geojson.Point((lon, lat))
        features.append(geojson.Feature(geometry=point))
    feature_collection = geojson.FeatureCollection(features)

    # Convert the GeoJSON to a string
    geojson_str = geojson.dumps(feature_collection)

    # Extract the timestamp from the dataset
    timestamp = ds.t.values

    return timestamp, geojson_str

def upload_to_bigquery(prediction_time, goesmask_geojson):
    """
    Uploads a row to a specified BigQuery table. The row contains a prediction time and a GeoJSON string.

    :param prediction_time: The prediction time to include in the row.
    :param goesmask_geojson: The GeoJSON string to include in the row.
    :return: None. The function prints a message indicating whether the row was inserted successfully.
    """
    # Initialize a BigQuery client
    client = bigquery.Client()

    # Specify your dataset and table
    dataset_id = 'geojson_predictions'
    table_id = 'goes_mask'

    # Get the table
    table = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table)

    # Convert numpy.datetime64 to datetime and then to string for bigquery
    prediction_time = pd.to_datetime(str(prediction_time)).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Prepare the row to be inserted
    row = {
        'prediction_date': prediction_time,
        'goes_mask_geojson': goesmask_geojson,
        'datetime_added': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),  # Add current UTC time
    }

    # Insert the row
    errors = client.insert_rows_json(table, [row])

    # Check if any errors occurred
    if errors:
        print('Errors:', errors)
    else:
        print('Row inserted successfully.')


def GOES_GEOJSON_UPDATE(request):
    """
    Main function to update GeoJSON data. It fetches the most recent blob from a GOES-16 bucket,
    generates GeoJSON points from the blob's dataset, and uploads the GeoJSON to a BigQuery table.

    :param request: The HTTP request that triggered this function. Not used in this function.
    :return: A string indicating that the function executed successfully and an HTTP status code of 200.
    """
    # At the start of your cloud function
    lat_mapping, lon_mapping = get_lat_lon_mapping('firenet_reference', 'goes16_abi_conus_lat_lon.nc')
    # Use fsspec to create a file system
    fs = fsspec.filesystem('gcs')
    most_recent_blob = get_most_recent_blob_name()
    dataset = get_dataset(most_recent_blob, fs)  # Get the dataset for the most recent blob
    # Generate GeoJSON points from the dataset
    timestamp, geojson_str = generate_geojson_points(dataset, lat_mapping, lon_mapping)
    # Delete lat_mapping, lon_mapping as they are no longer needed
    del lat_mapping, lon_mapping
    # Upload the generated GeoJSON to BigQuery
    upload_to_bigquery(timestamp, geojson_str)

    return 'Function executed successfully', 200