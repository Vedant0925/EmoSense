def my_function(user_mood):
    

    import lyricsgenius
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from textblob import TextBlob
    import numpy as np
    import pickle
    import os
    # import pygame
    import threading
    import keyboard


    GENIUS_API_KEY = "DyZvAtC52zoMh8uy90ZHZ2RGnIaxpGqVLsMwAjwfWrP7UhkVEJFv-5NPcYn4UXHy"
    SPOTIPY_CLIENT_ID = "aa087ea1788347d1a6a4b30cbd6fdd34"
    SPOTIPY_CLIENT_SECRET = "cac4f4e8e70f4a35b163665d533dd479"

    genius = lyricsgenius.Genius(GENIUS_API_KEY, timeout=15)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                            client_secret=SPOTIPY_CLIENT_SECRET))

    import time

    def get_lyrics(title, artist, max_retries=3, sleep_duration=2):
        retries = 0
        while retries < max_retries:
            try:
                song = genius.search_song(title, artist)
                if song:
                    return song.lyrics
                else:
                    return None
            except Exception as e:
                print(f"Error fetching lyrics for {title} by {artist}: {e}")
                retries += 1
                time.sleep(sleep_duration)

        print(f"Failed to fetch lyrics for {title} by {artist} after {max_retries} retries")
        return None



    def get_audio_features(track_id):
        return sp.audio_features([track_id])[0]


    def sentiment_score(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity


    import pandas as pd

    def create_dataset(playlist_id):
        tracks = sp.playlist_tracks(playlist_id)["items"]
        data = []

        for track in tracks:
            track_info = track["track"]
            track_id = track_info["id"]
            title = track_info["name"]
            artist = track_info["artists"][0]["name"]

            lyrics = get_lyrics(title, artist)
            if lyrics:
                audio_features = get_audio_features(track_id)
                sentiment = sentiment_score(lyrics)
                data.append({**audio_features, "sentiment": sentiment, "title": title, "artist": artist , "id": track_id})

        return pd.DataFrame(data)
    


    if user_mood.lower() == "sad" or user_mood.lower()=="bad":
        playlist_id = "4PWQV9dQpT7As9OTZBqrR8"
    else:
        playlist_id = "37i9dQZF1DXdPec7aLTmlC"
    # if language.lower()=='english':
    #     if user_mood.lower() == "sad" or user_mood.lower() == "bad":
    #         playlist_id = "4PWQV9dQpT7As9OTZBqrR8"
    #     else:
    #         playlist_id = "37i9dQZF1DXdPec7aLTmlC"


    data_file = f"{playlist_id}_data.pkl"

    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            df = pickle.load(f)
    else:
        df = create_dataset(playlist_id)
        with open(data_file, 'wb') as f:
            pickle.dump(df, f)



    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    def train_model(df, n_clusters=10):
        features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'sentiment']
        X = df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=n_clusters)
        model.fit(X_scaled)

        df['cluster'] = model.labels_

        return model, scaler


    def recommend_songs(df, model, scaler, mood, n_recommendations=10):
        mood_sentiment = sentiment_score(mood)

        mood_features = df.iloc[0][['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence']].values
        mood_features = list(mood_features)
        mood_features.append(mood_sentiment)
        mood_features = np.array([mood_features])


        mood_features_scaled = scaler.transform(mood_features)



        mood_cluster = model.predict(mood_features_scaled)

        cluster_songs = df[df['cluster'] == mood_cluster[0]]
        recommendations = cluster_songs.sample(min(n_recommendations, len(cluster_songs)))

        return recommendations[['title', 'id', 'artist']]



    model, scaler = train_model(df)

    recommended_songs = recommend_songs(df, model, scaler, user_mood)
    print("\nRecommended songs for your mood:")
    print(recommended_songs)

    return recommended_songs


from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/recommend_songs', methods=['POST'])
def recommend_songs():
    # Get the user mood from the POST request
    user_mood = request.json['user_mood']

    # Call the my_function with the user mood
    recommended_songs = my_function(user_mood)

    # Return the recommended songs as a JSON response
    return jsonify({'recommended_songs': recommended_songs.to_dict()}), 200

if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)

