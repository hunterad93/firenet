from sklearn.cluster import DBSCAN
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import numpy as np

#This .py holds the functions that takes mosaick and clusters fire booleans, draws polygons around them, uploads them as geojson strings to GBQ.


def stitch_chunks(tensors, original_shape):
    # Assuming processed_chunks is a list of 2D arrays (Height, Width)
    # and original_shape is the shape of the 2D plane of the original array (Height, Width)
    stitched = np.zeros(original_shape)
    processed_chunks = [t.detach().numpy().squeeze() for t in tensors]
    chunk_size = processed_chunks[0].shape[0]
    for k, chunk in enumerate(processed_chunks):
        i, j = divmod(k, original_shape[1] // chunk_size)
        stitched[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size] = chunk
    return stitched

def populate_template(template, stitched, data_var_index=0):
    data_var_name = list(template.data_vars)[data_var_index]
    template[data_var_name].values = np.expand_dims(stitched, axis=0)
    return template

def reproject_stitched_dataset(template):
    """
    Reprojects the given xarray Dataset to EPSG:4326, ensuring all data variables are compatible with NaN values.
    
    Parameters:
    - template: xarray.Dataset, the dataset to be reprojected.
    
    Returns:
    - xarray.Dataset, the reprojected dataset.
    """
    for var in template.data_vars:
        # Convert the data variable to float32 to ensure compatibility with NaN values
        template[var] = template[var].astype('float32')
        # Set the nodata value for the data variable
        template[var].rio.set_nodata(0, inplace=True)
    
    # Reproject the dataset to EPSG:4326
    reprojected_template = template.rio.reproject("EPSG:4326")
    
    return reprojected_template

def threshold_top_1_percent(dataset, threshold = 1):
    """
    Thresholds the given data variable in the dataset, setting the top 1% of values to 1 and the rest to 0.
    
    Parameters:
    - dataset: xarray.Dataset containing the data variable to be thresholded.
    - data_var_name: str, the name of the data variable within the dataset to apply thresholding to.
    
    Returns:
    - xarray.Dataset with the thresholded data variable.
    """

    # Apply thresholding: values above threshold set to 1, else set to 0
    first_data_var = list(dataset.data_vars)[0]
    thresholded_values = np.where(dataset[first_data_var].values > threshold, 1, 0)
    # Update the dataset with the thresholded values
    dataset[first_data_var].values = thresholded_values
    
    return dataset


def dataset_to_geojson_points(dataset, data_var_index = 0):
    """
    Converts 1s in the specified data variable of an xarray Dataset to GeoJSON points,
    utilizing the spatial metadata in the dataset.
    
    Parameters:
    - dataset: xarray.Dataset containing the data variable and spatial metadata.
    - data_var_name: str, the name of the data variable to convert to points.
    
    Returns:
    - GeoDataFrame: A GeoDataFrame containing the points.
    """

    
    # Extract the 2D array of the data variable
    data_array = dataset[list(dataset.data_vars)[data_var_index]].values
    data_array = data_array.squeeze()
    # Find indices where the value is 1
    y_indices, x_indices = np.where(data_array == 1)
    
    # Retrieve the geographic coordinates for each index
    x_coords = dataset.coords['x'][x_indices].values
    y_coords = dataset.coords['y'][y_indices].values
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(columns=['geometry'], geometry=[Point(xy) for xy in zip(x_coords, y_coords)])
    
    # Set the CRS from the dataset (assuming it's directly interpretable by GeoPandas)
    if hasattr(dataset, 'crs'):
        gdf.crs = dataset.crs
    
    return gdf

def cluster_fires(gdf, eps=0.01, min_samples=5):
    """
    Given a GeoDataFrame of fire points, create spatial clusters.
    This function extracts longitude and latitude from the 'geometry' column,
    performs DBSCAN clustering, and adds a 'label' column indicating the cluster each point belongs to.
    
    Parameters:
    - gdf: GeoDataFrame of fire points with a 'geometry' column of Point objects.
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    - GeoDataFrame of fire points with an additional column 'label' indicating the cluster each point belongs to.
    """
    # Extract longitude and latitude from the 'geometry' column
    gdf['longitude'] = gdf.geometry.x
    gdf['latitude'] = gdf.geometry.y

    # Perform DBSCAN clustering
    coords = gdf[['longitude', 'latitude']].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)

    # Add cluster labels to the dataframe
    gdf['label'] = db.labels_
      # Filter out noise points
    gdf = gdf[gdf['label'] != -1]

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


    return polygon_geojson