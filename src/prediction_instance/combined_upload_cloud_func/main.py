from datetime import datetime, timedelta
from io import StringIO
import geopandas as gpd
import requests
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
import json
import simplekml
from google.cloud import bigquery, storage

def get_firms_data(api_key, bbox, product, days_of_data = 2, date=None):
    '''
    Connect with FIRMS API to access data from a specified date, bbox, product, and range of days
    and return it as a GeoDataFrame. If no date is specified, defaults to today.
    
    :param api_key: str, from NASA email, provided in cron job's request headers
    :param bbox: str, bbox of the region of interest in the format "minLongitude,minLatitude,maxLongitude,maxLatitude", provided in cron job's request headers
    :param date: str, date in '%Y-%m-%d' format. If not provided, defaults to today.
    :return: GeoDataFrame of fire detection data with columns corresponding to the FIRMS API response
    '''
    
    base_url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/'

    # Request `days_of_data` worth of data, before filtering via the acq_date/time
    url = f'{base_url}{api_key}/{product}/{bbox}/{days_of_data}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the request was unsuccessful
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching data: {e}")
    else:
        data = StringIO(response.text)  # Convert text response to file-like object
        df = pd.read_csv(data)  # Read data into a DataFrame


    # Convert the DataFrame to a GeoDataFrame, setting the geometry from the latitude and longitude columns
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    # Drop unnecessary columns
    columns_to_keep = ['latitude', 'longitude', 'confidence', 'geometry', 'acq_date', 'acq_time']
    gdf = gdf[columns_to_keep]

    # Add a column indicating the product
    gdf['product'] = product

    return gdf

def filter_by_datetime(gdf, days=1):
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
    print(len(gdf))
    # Get the latest time in the GeoDataFrame
    latest_time = gdf['datetime'].max()
    print(latest_time)
    # Get the time 24 hours before the latest time
    one_day_before_latest = latest_time - pd.Timedelta(days=days)

    # Filter rows from the last 24 hours based on the latest time
    gdf = gdf[gdf['datetime'] >= one_day_before_latest]
    print(len(gdf))
    return gdf


def convert_landsat_confidence_column(landsat_gdf):
    """
    Convert the 'confidence' column in a Landsat GeoDataFrame to the same system as VIIRS.
    :param landsat_gdf: GeoDataFrame containing Landsat fire data with a 'confidence' column
    :return: GeoDataFrame with the 'confidence' column converted to categorical values (l='low', n='nominal', h='high')
    """
    def categorize_confidence(value):
        if value == 'H':
            return 'h'
        elif value == 'M':
            return 'n'
        else:  # Assuming any value not 'H' or 'M' is 'L'
            return 'l'

    landsat_gdf['confidence'] = landsat_gdf['confidence'].apply(categorize_confidence)
    return landsat_gdf


def convert_modis_confidence_column(modis_gdf):
    """
    Convert the 'confidence' column in a MODIS GeoDataFrame to categorical values based on confidence level.
    This changes the confidence categorization to match VIIRS
    :param modis_gdf: GeoDataFrame containing MODIS fire data with a 'confidence' column
    :return: GeoDataFrame with the 'confidence' column converted to categorical values ('low', 'nominal', 'high')
    """
    def categorize_confidence(value):
        if value > 50:
            return 'h'
        elif 25 < value <= 50:
            return 'n'
        else:
            return 'l'

    modis_gdf['confidence'] = modis_gdf['confidence'].apply(categorize_confidence)
    return modis_gdf

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

def filter_clusters(gdf, min_cluster_size, min_high_confidence):
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

def filter_clusters_with_product_confidence(gdf, min_cluster_size, required_high_confidence_per_product):
    """
    Filter out clusters that have fewer points than the threshold, and ensure at least one high confidence point
    from each product exists within the cluster.
    
    :param gdf: GeoDataFrame of fire points with 'label' column indicating the cluster each point belongs to
    :param min_cluster_size: Minimum number of points in a cluster for it to be kept
    :param required_high_confidence_per_product: Minimum number of high confidence points from each product in a cluster for it to be kept
    :return: GeoDataFrame of fire points in clusters that meet both thresholds
    """

    # Count the number of points in each cluster
    cluster_counts = gdf['label'].value_counts()

    # Filter out clusters smaller than the minimum size
    valid_clusters_by_size = cluster_counts[cluster_counts >= min_cluster_size].index

    # Filter for high confidence points
    high_confidence_gdf = gdf[gdf['confidence'] == 'h']

    # Ensure at least one high confidence point from each product within the cluster
    valid_clusters_by_product_confidence = high_confidence_gdf.groupby('label')['product'].nunique()
    valid_clusters_by_product_confidence = valid_clusters_by_product_confidence[valid_clusters_by_product_confidence >= required_high_confidence_per_product].index

    # Find the intersection of clusters that meet both criteria
    valid_clusters = set(valid_clusters_by_size) & set(valid_clusters_by_product_confidence)

    # Filter the GeoDataFrame to include only valid clusters
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
    most_common_acq_date = pd.to_datetime(gdf['datetime'].mode()[0])

    return most_common_acq_date, polygon_geojson

def convert_geojson_to_kml_and_upload(geojson_string, most_common_acq_date, bucket_name='popex_active_fire_kmls'):
    # Convert the GeoJSON string to a Python dictionary
    geojson = json.loads(geojson_string)
    
    # Create a new KML object
    kml = simplekml.Kml()
    
    # Iterate over each feature in the GeoJSON and add it to the KML
    for feature in geojson.get('features', []):
        geometry_type = feature['geometry']['type']
        coordinates = feature['geometry']['coordinates']
        
        if geometry_type == 'Polygon':
            pol = kml.newpolygon(name=feature.get('id', 'No ID'))
            pol.outerboundaryis = coordinates[0]  # Assuming no holes
        
    kml_file_name = f"firms_polygons_{most_common_acq_date}.kml"
    
    # Save the KML to a temporary file
    temp_kml_path = f"/tmp/{kml_file_name}"
    kml.save(temp_kml_path)
    
    # Upload the KML file to the specified Google Cloud Storage bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(kml_file_name)
    
    blob.upload_from_filename(temp_kml_path)
    
    print(f"Uploaded {kml_file_name} to {bucket_name}.")

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
    table_id = 'combined_firms_mask'

    # Get the table
    table = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table)

    # Convert acq_date to string for bigquery
    acq_date = acq_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Prepare the row to be inserted
    row = {
        'prediction_date': acq_date,
        'geojson_mask': polygon_geojson,
        'datetime_added': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),  # UTC timestamp of the current moment
    }

    # Insert the row
    errors = client.insert_rows_json(table, [row])

    # Check if any errors occurred
    if errors:
        print('Errors:', errors)
    else:
        print('Row inserted successfully.')

def FIRMS_GEOJSON_UPDATE(request):
    # Extract request parameters from the cron job request
    request_json = request.get_json(silent=True)
    api_key = request_json.get('api_key')
    bbox = request_json.get('bbox', 'world')
    products = request_json.get('products', ["VIIRS_SNPP_NRT", "VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"])
    min_cluster_size = request_json.get('min_cluster_size', 40)  # Default value set to 40 if not specified
    required_high_confidence = request_json.get('required_high_confidence', 3)  # Default value set to 3 if not specified

    # Retrieve data using the provided API key, bounding box, and list of products
    gdfs = [get_firms_data(api_key=api_key, bbox=bbox, product=product) for product in products]
    gdfs = [filter_by_datetime(gdf) for gdf in gdfs]
    # Apply product-specific confidence conversion if MODIS product is in the list
    if "MODIS_NRT" in products:
        modis_index = products.index("MODIS_NRT")
        gdfs[modis_index] = convert_modis_confidence_column(gdfs[modis_index])

    combined_gdf = pd.concat(gdfs, ignore_index=True)
    # Cluster the combined data points
    clustered_combined_gdf = cluster_fires(combined_gdf)
    # Filter out small clusters and clusters with too few points or no high confidence point
    filtered_combined_clusters = filter_clusters_with_product_confidence(clustered_combined_gdf, min_cluster_size=min_cluster_size, required_high_confidence_per_product=required_high_confidence)
    # Create a polygon for each cluster
    acq_date, polygon_geojson = create_cluster_polygons(filtered_combined_clusters)

    # Upload as KML before uploading to GBQ
    convert_geojson_to_kml_and_upload(polygon_geojson,acq_date)

    # Upload the generated polygon GeoJSON to BigQuery
    upload_to_bigquery(acq_date, polygon_geojson)

    return 'Successfully processed and uploaded data', 200