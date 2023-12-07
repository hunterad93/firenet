from flask import Flask, render_template
from flask import send_file
from utils import download_blob
from flask import Response
import os



app = Flask(__name__)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/adamhunter/Documents/school projs/firenet/data/credentials/firenet-99-reader.json'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/geojson')
def serve_geojson():
    geojson_data = download_blob('prediction_imagery', 'predictions.geojson')
    return Response(geojson_data, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, port=5000)

