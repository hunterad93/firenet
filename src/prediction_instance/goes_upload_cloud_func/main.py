
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

def get_blob_names(attime=datetime.utcnow(), within=pd.to_timedelta("20min"), bucket_name='gcp-public-data-goes-16'):
    if isinstance(attime, str):
        attime = pd.to_datetime(attime)
    if isinstance(within, str):
        within = pd.to_timedelta(within)

    # Parameter Setup
    start = attime - within
    end = attime + within

    # Set up Google Cloud Storage client
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Create a range of directories to check. The GOES bucket is
    # organized by hour of day.
    blob_names = []
    current_time = start
    while current_time <= end:
        prefix = f'ABI-L2-FDCC/{current_time.year}/{current_time.timetuple().tm_yday:03d}/{current_time.hour:02d}/'
        blobs = bucket.list_blobs(prefix=prefix)
        blob_names.extend([blob.name for blob in blobs])
        current_time += timedelta(minutes=20)  # Increment current_time by 20 minutes

    return blob_names

def select_blobs(blob_names):
    # Sort blob names by timestamp
    blob_names.sort(key=lambda name: name.split('_')[3][1:], reverse=True)  # Extract timestamp after 's'

    # Extract band numbers from blob names
    try:
        band_numbers = [int(name.split('_')[1][-2:]) for name in blob_names]
    except ValueError:
        # If there is only one band and the band number is a string, return the most recent file
        return [blob_names[0]]

    # Get unique band numbers and sort them
    unique_band_numbers = sorted(set(band_numbers))

    # If there is only one unique band number, return the most recent file
    if len(unique_band_numbers) == 1:
        return [blob_names[0]]

    # Find the first continuous sequence that matches the expected band order
    for i in range(len(blob_names) - len(unique_band_numbers) + 1):
        selected = blob_names[i:i+len(unique_band_numbers)]
        band_order = [int(name.split('_')[1][-2:]) for name in selected]
        if band_order == unique_band_numbers:
            break
    else:
        raise Exception("No continuous sequence found that matches the expected band order")
    return selected

def get_datasets(blob_names, fs, bucket_name='gcp-public-data-goes-16'):
    # Open each blob as an xarray Dataset and store it in the dictionary under the corresponding channel identifier
    datasets = {}
    for name in blob_names:
        channel_id = name.split('_')[1]
        f = fs.open(f'{bucket_name}/{name}')
        ds = xr.open_dataset(f, engine='h5netcdf')
        datasets[channel_id] = ds
    return datasets

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
    blob_names = get_blob_names()
    selected_blobs = select_blobs(blob_names)
    datasets = get_datasets(selected_blobs, fs)
    if datasets:
        first_ds_key = next(iter(datasets))
    else:
        # Handle the empty case, perhaps log an error or return
        print("No datasets available.")
        return

    first_ds = datasets[first_ds_key]

    # Generate GeoJSON points from the first dataset
    timestamp, geojson_str = generate_geojson_points(first_ds, lat_mapping, lon_mapping)
    # Delete lat_mapping, lon_mapping as they are no longer needed
    del lat_mapping, lon_mapping
    # Upload the generated GeoJSON to BigQuery
    upload_to_bigquery(timestamp, geojson_str)

    return 'Function executed successfully', 200