# Fire Prediction Processing

## Overview
This repository contains cloud functions responsible for fetching, preprocessing, and uploading fire prediction data from VIIRS, GOES, and Firenet to Google BigQuery (GBQ) in geojson format. It additionally includes the code for a cloud function which generates Firenet predictions and stores them as numpy arrays in a google cloud bucket.

Find more info, and view the map at the projects website: 
https://sites.google.com/view/firenet-/home

## Data Sources
The raw data is sourced from the following locations:
- VIIRS: [VIIRS Data](https://firms.modaps.eosdis.nasa.gov/usfs/api/area/)
- GOES: [GOES Data](https://console.cloud.google.com/storage/browser/gcp-public-data-goes-16)

## Requirements
This project requires Python 3.9 and the packages listed in `requirements.txt`.

## Scratch work
The scratch work folder includes the work of the data scientist Sean Carter who originally developed firenet, it also includes work by me (Adam Hunter) that was refactored into the production ready cloud functions described below.

## Prediction Instance Cloud Functions
The cloud functions are defined in the folders named in accordance with their prediction sources. `goes_upload_cloud_func` includes code for the cloud function that uploads GOES predictions. `viirs_upload_cloud_func` includes the code for the cloud function that uploads VIIRS predictions. `unet_upload_cloud_func` includes the code for firenet upload.