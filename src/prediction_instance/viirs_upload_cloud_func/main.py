from datetime import datetime, timedelta
from io import StringIO
import geopandas as gpd
import requests
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from google.cloud import bigquery

def get_viirs_data(api_key, bbox, date=None):
    '''
    Connect with FIRMS API to access VIIRS detection data from a specified date and the day before
    and return it as a GeoDataFrame. If no date is specified, defaults to today.
    
    :param api_key: str, from NASA email
    :param bbox: str, bbox of the region of interest in the format "minLongitude,minLatitude,maxLongitude,maxLatitude"
    :param date: str, date in '%Y-%m-%d' format. If not provided, defaults to today.
    :return: GeoDataFrame of VIIRS detection data with columns corresponding to the FIRMS API response
    '''
    
    base_url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/'

    # If no date is provided, default to today's date
    if date is None:
        date = datetime.now()
    else:
        date = datetime.strptime(date, '%Y-%m-%d')

    # Get the date for the day before
    day_before = date - timedelta(days=1)

    # Format dates to '%Y-%m-%d' and create a list of dates from the day before to the specified date
    dates = [day_before.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')]

    data_frames = []  # List to store data frames

    for date in dates:
        url = f'{base_url}{api_key}/VIIRS_SNPP_NRT/{bbox}/1/{date}'
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception if the request was unsuccessful
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while fetching data: {e}")
            continue

        data = StringIO(response.text)  # Convert text response to file-like object

        df = pd.read_csv(data)  # Read data into a DataFrame
        data_frames.append(df)

    # Concatenate all data frames into one
    viirs_data = pd.concat(data_frames, ignore_index=True)

    # Convert the DataFrame to a GeoDataFrame, setting the geometry from the latitude and longitude columns
    gdf = gpd.GeoDataFrame(viirs_data, geometry=gpd.points_from_xy(viirs_data.longitude, viirs_data.latitude))

    # Drop unnecessary columns
    columns_to_keep = ['latitude', 'longitude', 'confidence', 'geometry', 'acq_date', 'acq_time']
    gdf = gdf[columns_to_keep]

    return gdf

def filter_last_24_hours(gdf):
    """
    Filter the GeoDataFrame to include only rows from the last 24 hours.
    
    :param gdf: GeoDataFrame with 'acq_date' and 'acq_time' columns
    :return: GeoDataFrame with rows from the last 24 hours
    """
    # Convert 'acq_time' to a string and pad it with zeros to ensure it has four digits
    gdf['acq_time'] = gdf['acq_time'].astype(str).str.zfill(4)

    # Extract the hours and minutes from 'acq_time'
    gdf['hour'] = gdf['acq_time'].str[:2]
    gdf['minute'] = gdf['acq_time'].str[2:]

    # Combine 'acq_date', 'hour', and 'minute' into a single datetime column
    gdf['datetime'] = pd.to_datetime(gdf['acq_date'] + ' ' + gdf['hour'] + ':' + gdf['minute'])

    # Sort the GeoDataFrame by 'datetime'
    gdf = gdf.sort_values('datetime')

    # Get the latest time in the GeoDataFrame
    latest_time = gdf['datetime'].max()

    # Get the time 24 hours before the latest time
    one_day_before_latest = latest_time - pd.Timedelta(days=1)

    # Filter rows from the last 24 hours based on the latest time
    gdf = gdf[gdf['datetime'] >= one_day_before_latest]

    return gdf

def cluster_fires(gdf, eps=0.01, min_samples=1):
    """
    Given a GeoDataFrame of fire points, create spatial clusters
    :param gdf: GeoDataFrame of fire points
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
    :return: GeoDataFrame of fire points with an additional column 'label' indicating the cluster each point belongs to
    """

    # Perform DBSCAN clustering
    coords = gdf[['longitude', 'latitude']].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)

    # Add cluster labels to the dataframe
    gdf['label'] = db.labels_

    return gdf

def filter_clusters(gdf, min_cluster_size=10, min_high_confidence=1):
    """
    Filter out clusters that have fewer points, and fewer high confidence points, than the two thresholds
    :param gdf: GeoDataFrame of fire points with 'label' column indicating the cluster each point belongs to
    :param min_cluster_size: Minimum number of points in a cluster for it to be kept
    :param min_high_confidence: Minimum number of high confidence points in a cluster for it to be kept
    :return: GeoDataFrame of fire points in clusters that meet both thresholds
    """

    # Count the number of points in each cluster
    cluster_counts = gdf['label'].value_counts()

    # Count the number of high confidence points in each cluster
    high_confidence_counts = gdf.loc[gdf['confidence'] == 'h']['label'].value_counts()

    # Filter out small clusters and clusters with too few high confidence points
    valid_clusters = cluster_counts[(cluster_counts >= min_cluster_size) & (high_confidence_counts >= min_high_confidence)].index
    gdf = gdf[gdf['label'].isin(valid_clusters)]

    return gdf

def create_cluster_polygons(gdf):
    """
    Given a GeoDataFrame of clustered fire points, create a polygon for each cluster
    :param gdf: GeoDataFrame of fire points with 'label' column indicating the cluster each point belongs to
    :return: Tuple containing the most frequently occurring acquisition date and a GeoJSON string where each feature represents a cluster and the geometry property contains the polygon around the cluster
    """
    # Group the GeoDataFrame by the cluster labels
    grouped = gdf.groupby('label')

    # For each cluster, create a MultiPoint object from the fire points, then create a polygon from the convex hull of the points
    polygons = grouped.apply(lambda df: MultiPoint(df.geometry.tolist()).convex_hull)

    # Create a new GeoDataFrame from the polygons
    polygon_gdf = gpd.GeoDataFrame({'geometry': polygons})

    # Convert the GeoDataFrame to a GeoJSON string
    polygon_geojson = polygon_gdf.to_json()

    # Convert the most frequently occurring acquisition date to datetime
    most_common_acq_date = pd.to_datetime(gdf['acq_date'].mode()[0])

    return most_common_acq_date, polygon_geojson

def upload_to_bigquery(acq_date, polygon_geojson):
    """
    Uploads the polygon GeoJSON data to BigQuery.

    :param acq_date: The most frequently occurring acquisition date. There will only ever be two dates in the GDF.
    :param polygon_geojson: The GeoJSON string where each feature represents a cluster and the geometry property contains the polygon around the cluster.
    """
    # Initialize a BigQuery client
    client = bigquery.Client()

    # Specify your dataset and table
    dataset_id = 'geojson_predictions'
    table_id = 'viirs_mask'

    # Get the table
    table = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table)

    # Convert acq_date to string for bigquery
    acq_date = acq_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Prepare the row to be inserted
    row = {
        'prediction_date': acq_date,
        'viirs_mask_geojson': polygon_geojson,
        'datetime_added': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),  # UTC timestamp of the current moment
    }

    # Insert the row
    errors = client.insert_rows_json(table, [row])

    # Check if any errors occurred
    if errors:
        print('Errors:', errors)
    else:
        print('Row inserted successfully.')

def VIIRS_GEOJSON_UPDATE(request):
    # Get the request parameters from the cron job request that is sent to the cloud funtion
    # The GCP cron job is where the API key and bbox are specified
    request_json = request.get_json(silent=True)

    api_key = request_json['api_key']
    bbox = request_json['bbox']
    # Delete request_json as it's no longer needed
    del request_json

    # Get the VIIRS data
    viirs_data = get_viirs_data(api_key, bbox)

    # Filter out points from the last 24 hours
    viirs_data = filter_last_24_hours(viirs_data)

    # Cluster the fire points
    clustered_fires = cluster_fires(viirs_data)
    # Delete viirs_data as it's no longer needed
    del viirs_data

    # Filter out small clusters and clusters with too few high confidence points
    filtered_clusters = filter_clusters(clustered_fires)
    # Delete clustered_fires as it's no longer needed
    del clustered_fires

    # Create a polygon for each cluster
    acq_date, polygon_geojson = create_cluster_polygons(filtered_clusters)
    # Delete filtered_clusters as it's no longer needed
    del filtered_clusters

    # Upload the polygon to BigQuery
    upload_to_bigquery(acq_date, polygon_geojson)
    # Delete acq_date and polygon_geojson as they're no longer needed
    del acq_date, polygon_geojson

    return 'Successfully processed and uploaded data', 200
