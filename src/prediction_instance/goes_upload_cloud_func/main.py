
import fsspec
import xarray as xr
import numpy as np
import geojson
from google.cloud import storage
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery

def get_lat_lon_mapping(bucket_name, blob_name):
    fs = fsspec.filesystem('gcs')
    with fs.open(f'{bucket_name}/{blob_name}') as f:
        ds = xr.open_dataset(f)
        lat_mapping = ds['latitude'].values.astype(float)
        lon_mapping = ds['longitude'].values.astype(float)
    return lat_mapping, lon_mapping

def get_most_recent_blob_name(bucket_name='gcp-public-data-goes-16'):
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
    # Open the blob as an xarray Dataset
    f = fs.open(f'{bucket_name}/{blob_name}')
    ds = xr.open_dataset(f, engine='h5netcdf')
    return ds

def generate_geojson_points(ds, lat_mapping, lon_mapping):
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
    # Initialize a BigQuery client
    client = bigquery.Client()

    # Specify your dataset and table
    dataset_id = 'geojson_predictions'
    table_id = 'goesmask'

    # Get the table
    table = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table)

    # Convert numpy.datetime64 to datetime and then to string for bigquery
    prediction_time = pd.to_datetime(str(prediction_time)).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Prepare the row to be inserted
    row = {
        'prediction_date': prediction_time,
        'goesmask_geojson': goesmask_geojson,
    }

    # Insert the row
    errors = client.insert_rows_json(table, [row])

    # Check if any errors occurred
    if errors:
        print('Errors:', errors)
    else:
        print('Row inserted successfully.')


def GOES_GEOJSON_UPDATE(request):
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