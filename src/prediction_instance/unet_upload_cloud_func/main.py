import preprocessing as pre
import neuralnet_processing as nn
import postprocessing as post
from google.cloud import bigquery
from datetime import datetime
import gc #using this to manually free resources mid-function

def preprocess_data():
    print('preprocessing')
    # Creates a file system object for interacting with GCP as if it were a local file system
    fs = pre.create_fs()
    # Downloads and processes the selected blobs in parallel to create a median image dataset
    stacked_ds = pre.download_blob(fs)

    # Creates a spatial template from the dataset, preserving spatial metadata but with NaN values
    template = pre.create_spatial_template(stacked_ds)
    # Extracts the data variables as a 3D NumPy array from the xarray dataset
    npy_array = pre.extract_data_as_array(stacked_ds)
    del stacked_ds  # Free up memory
    gc.collect()
    # Breaks down the NumPy array into smaller chunks for processing or model input
    chunks = pre.chunk_ndarray(npy_array)
    return template, chunks, npy_array.shape[-2:]

def run_neuralnet(chunks):
    print('net initiated')
    # Load the trained U-Net model from the specified bucket and blob
    unet = nn.load_unet_model()
    
    # Predict on the chunks
    prediction_tensors = nn.predict(chunks, unet)
    
    return prediction_tensors

def postprocess_data(prediction_tensors, original_shape, template, eps=0.01, min_samples=5, threshold=3):
    print('postprocessing')
    # Stitch the processed chunks together
    stitched = post.stitch_chunks(prediction_tensors, original_shape)
    
    # Populate the template dataset with the stitched data
    populated_template = post.populate_template(template, stitched)
    
    # Reproject the dataset to EPSG:4326
    reprojected_dataset = post.reproject_stitched_dataset(populated_template)
    
    # Apply thresholding to the dataset
    thresholded_dataset = post.threshold_top_1_percent(reprojected_dataset, threshold)
    
    # Convert the dataset to GeoJSON points
    gdf_points = post.dataset_to_geojson_points(thresholded_dataset)
    
    # Cluster the fire points
    clustered_gdf = post.cluster_fires(gdf_points, eps, min_samples)
    
    # Create polygons for each cluster
    cluster_polygons_geojson = post.create_cluster_polygons(clustered_gdf)
    
    return cluster_polygons_geojson

def upload_to_bigquery(polygon_geojson):
    """
    Uploads the polygon GeoJSON data to BigQuery.

    :param polygon_geojson: The GeoJSON string where each feature represents a cluster and the geometry property contains the polygon around the cluster.
    """
    # Initialize a BigQuery client
    client = bigquery.Client()

    # Specify your dataset and table
    dataset_id = 'geojson_predictions'
    table_id = 'unet'

    # Get the table
    table = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table)

    # Prepare the row to be inserted
    row = {
        'prediction_date': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),  # UTC timestamp of the current moment
        'unet_geojson': polygon_geojson
    }

    # Insert the row
    errors = client.insert_rows_json(table, [row])

    # Check if any errors occurred
    if errors:
        print('Errors:', errors)
    else:
        print('Row inserted successfully.')

def UNET_GEOJSON_UPDATE(request):
    """
    Orchestrates the process of updating the UNET model predictions as GeoJSON in BigQuery without requiring external inputs.
    The model bucket name and blob name are defined within the function.
    """

    # Preprocess data
    template, chunks, original_shape = preprocess_data()

    # Run neural network predictions
    prediction_tensors = run_neuralnet(chunks)

    # Postprocess the data
    polygon_geojson = postprocess_data(prediction_tensors, original_shape, template)

    # Upload to BigQuery
    upload_to_bigquery(polygon_geojson)

    return 'GeoJSON updated successfully in BigQuery.'

# Leave as is for cloud function
# UNET_GEOJSON_UPDATE(0) #uncomment this for local testing