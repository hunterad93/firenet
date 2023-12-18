
import os
import fsspec
import xarray as xr
import numpy as np
import geojson
from google.cloud import storage
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery

def calculate_degrees(file_id):
    
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables['x'][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables['y'][:]  # N/S elevation angle in radians
    projection_info = file_id.variables['goes_imager_projection']
    lon_origin = projection_info.attrs['longitude_of_projection_origin']
    H = projection_info.attrs['perspective_point_height'] + projection_info.attrs['semi_major_axis']
    r_eq = projection_info.attrs['semi_major_axis']
    r_pol = projection_info.attrs['semi_minor_axis']    
    
    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
    
    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin*np.pi)/180.0  
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)
    
    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all='ignore')
    
    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return abi_lat, abi_lon

def get_blob_names(attime=datetime.utcnow(), within=pd.to_timedelta("1H"), bucket_name='gcp-public-data-goes-16'):
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
    for hour in range(start.hour, end.hour + 1):
        prefix = f'ABI-L2-FDCC/{start.year}/{start.timetuple().tm_yday:03d}/{hour:02d}/'
        blobs = bucket.list_blobs(prefix=prefix)
        blob_names.extend([blob.name for blob in blobs])

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

def generate_geojson_points(ds):
    # Process the data to generate GeoJSON
    band_data = ds['DQF'].values
    zero_indices = np.where(band_data == 0)
    lat, lon = calculate_degrees(ds)
    zero_lat_lon = lat[zero_indices], lon[zero_indices]
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


def main(event, context):
    # Use fsspec to create a file system
    fs = fsspec.filesystem('gcs', token=os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    blob_names = get_blob_names()
    selected_blobs = select_blobs(blob_names)
    datasets = get_datasets(selected_blobs, fs)
    # Extract the first dataset from the datasets dictionary
    first_ds_key = next(iter(datasets))
    first_ds = datasets[first_ds_key]

    # Generate GeoJSON points from the first dataset
    timestamp, geojson_str = generate_geojson_points(first_ds)
    # Upload the generated GeoJSON to BigQuery
    upload_to_bigquery(timestamp, geojson_str)