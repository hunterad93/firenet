# Video Overview 🎥


[![YouTube video player](https://img.youtube.com/vi/DTYgxgTlXcU/0.jpg)](https://www.youtube.com/watch?v=hF4GIpFAo24)
## What is Firenet?

Firenet is a state-of-the-art neural network model designed to enhance wildfire detection and monitoring capabilities by leveraging the strengths of both geostationary and low Earth orbit satellite data. While geostationary satellites like GOES provide continuous monitoring with high temporal resolution, they are limited by lower spatial resolution. Conversely, the VIIRS satellite offers high spatial resolution but with lower temporal frequency, revisiting the same area only every 12 hours.

Firenet bridges this gap by being trained on VIIRS data and using GOES data as its input, allowing it to deliver predictions with the temporal resolution of GOES and a level of accuracy that aims to match or surpass that of VIIRS. This synergy enables Firenet to provide real-time fire predictions even in instances where cloud cover may obstruct the view of VIIRS, ensuring continuous monitoring capabilities. For example, if VIIRS misses a fire due to cloud cover, it would not be able to detect it until its next orbit, whereas Firenet can classify the fire as soon as the clouds dissipate, utilizing the persistent observation advantage of GOES.

The primary goal of Firenet is to demonstrate the effectiveness of integrating diverse satellite data into a neural network model for improved fire detection and prediction. With sufficient resources for ongoing development, Firenet has the potential to become an indispensable tool for firefighting agencies, offering early warnings and supporting quicker response times and more effective fire management strategies.

## Repository Overview
This repository contains cloud functions responsible for fetching, preprocessing, and uploading fire prediction data from VIIRS, GOES, and Firenet to Google BigQuery (GBQ) in geojson format. It additionally includes the code for a cloud function which generates Firenet predictions and stores them as numpy arrays in a google cloud bucket.

Find more info, and view the map at the projects website: 
https://sites.google.com/view/firenet-/home

## Data Sources
The raw data is sourced from the following locations:
- VIIRS: [VIIRS Data](https://firms.modaps.eosdis.nasa.gov/usfs/api/area/) This is the training data from the low earth orbit satellite.
- GOES: [GOES Data](https://console.cloud.google.com/storage/browser/gcp-public-data-goes-16) This is the model input from the geostationary satellite.
- LANDFIRE: [LANDFIRE Data](https://landfire.gov/version_download.php) These are static layers produced by The Department of Forestry, used as model input and include information on elevation, vegetation height, vegetation density, and other layers relating to burning potential.


## Requirements
This project requires Python 3.9 and the packages listed in `requirements.txt`.

## Scratch work
The scratch work folder includes the work of the data scientist Sean Carter who originally developed firenet, it also includes work by me (Adam Hunter) that was refactored into the production ready cloud functions described below.

## Prediction Instance Cloud Functions
The cloud functions are defined in the folders named in accordance with their prediction sources. `goes_upload_cloud_func` includes code for the cloud function that uploads GOES predictions. `viirs_upload_cloud_func` includes the code for the cloud function that uploads VIIRS predictions. `unet_upload_cloud_func` includes the code for firenet upload.

## Reporting Document
Check out `capstone_draft.docx` for the most recent draft of my capstone writeup.