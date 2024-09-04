import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the models and data
model = pickle.load(open('artifacts/model.pkl', 'rb'))
place_names = pickle.load(open('artifacts/place_names.pkl', 'rb'))
ratings_with_places = pickle.load(open('artifacts/ratings_with_places.pkl', 'rb'))
places_pivot = pickle.load(open('artifacts/places_pivot.pkl', 'rb'))
places_df = pd.read_csv('data/places.csv')  # Load your places CSV

def fetch_poster(suggestion):
    place_name = []
    ids_index = []
    poster_url = []

    for place_id in suggestion:
        place_name.append(places_pivot.index[place_id])

    for name in place_name[0]: 
        ids = np.where(ratings_with_places['PlaceName'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = ratings_with_places.iloc[idx]['ImageURL']
        poster_url.append(url)

    return poster_url

def recommend_place(place_name):
    places_list = []
    place_id = np.where(places_pivot.index == place_name)[0][0]
    distance, suggestion = model.kneighbors(places_pivot.iloc[place_id,:].values.reshape(1,-1), n_neighbors=6 )

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
            places = places_pivot.index[suggestion[i]]
            for j in places:
                places_list.append(j)
    return places_list, poster_url

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if 'place_name' not in data:
        return jsonify({'error': 'place_name is required'}), 400

    place_name = data['place_name']
    recommended_places, poster_url = recommend_place(place_name)
    response = {
        'recommended_places': recommended_places[1:],  # Skip the first one as it's the queried place itself
        'poster_url': poster_url[1:]  # Skip the first one as it's the queried place itself
    }
    return jsonify(response)

@app.route('/places', methods=['GET'])
def get_places():
    places_list = places_df['PlaceName'].tolist()
    return jsonify(places_list)

@app.route('/place/<place_name>', methods=['GET'])
def get_place_details(place_name):
    place = places_df[places_df['PlaceName'] == place_name].iloc[0]
    place_details = {
        'name': place['PlaceName'],
        'image': place['ImageURL'],
        'description': place.get('Description', 'No description available.')
    }
    return jsonify(place_details)

# Health Check Route
@app.route('/status', methods=['GET'])
def status():
    return jsonify({'message': 'Service is operational'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Get the port from environment variable
    app.run(host='0.0.0.0', port=port, debug=True)
