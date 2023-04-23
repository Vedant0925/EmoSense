    

def my_function(user_mood):


    import lyricsgenius
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from spotipy.oauth2 import SpotifyOAuth
    from textblob import TextBlob
    import numpy as np
    import pickle
    import os
    import pygame
    import threading
    import keyboard



    GENIUS_API_KEY = "DyZvAtC52zoMh8uy90ZHZ2RGnIaxpGqVLsMwAjwfWrP7UhkVEJFv-5NPcYn4UXHy"
    SPOTIPY_CLIENT_ID = "4c8e8472ba464245a39b3d3e9a11dcf4"
    SPOTIPY_CLIENT_SECRET = "eb996f580db84f44b52b44c9450e7036"

    genius = lyricsgenius.Genius(GENIUS_API_KEY, timeout=15)

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="4c8e8472ba464245a39b3d3e9a11dcf4",
                                                client_secret="eb996f580db84f44b52b44c9450e7036",
                                                redirect_uri="https://localhost:8000/callback",
                                                scope="user-library-read user-top-read"))

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

    def get_user_top_tracks(sp, limit=50, time_range='medium_term'):
        return sp.current_user_top_tracks(limit=limit, time_range=time_range)['items']


    def sentiment_score(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity


    import pandas as pd


    def get_genre_tracks(genre, limit=50):
        result = sp.search(q=f'genre:"{genre}"', limit=limit, type='track')
        tracks = result['tracks']['items']
        track_ids = [track['id'] for track in tracks]
        return track_ids


    def create_dataset(genres):
        data = []
        for genre in genres:
            track_ids = get_genre_tracks(genre)

            for track_id in track_ids:
                track_info = sp.track(track_id)
                title = track_info["name"]
                artist = track_info["artists"][0]["name"]

                lyrics = get_lyrics(title, artist)
                if lyrics:
                    audio_features = get_audio_features(track_id)
                    sentiment = sentiment_score(lyrics)
                    data.append({**audio_features, "sentiment": sentiment, "title": title, "artist": artist ,  "id": track_id})

        return pd.DataFrame(data)

    def create_dataset_from_playlist(playlist_id):
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
                data.append({**audio_features, "sentiment": sentiment, "title": title, "artist": artist})

        return pd.DataFrame(data)




    genres_to_search = ['rock', 'pop', 'jazz', 'electronic', 'hip hop', 'classical']
    data_file = "multi_genre_data.pkl"

    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            df = pickle.load(f)
    else:
        df = create_dataset(genres_to_search)
        with open(data_file, 'wb') as f:
            pickle.dump(df, f)

    # The rest of your existing code remains the same
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


    def recommend_songs(source_df, model, scaler, mood, n_recommendations=10):
        mood_sentiment = sentiment_score(mood)

        mood_features = df.iloc[0][['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence']].values
        mood_features = list(mood_features)
        mood_features.append(mood_sentiment)
        mood_features = np.array([mood_features])


        mood_features_scaled = scaler.transform(mood_features)



        mood_cluster = model.predict(mood_features_scaled)

        cluster_songs = df[df['cluster'] == mood_cluster[0]]
        recommendations = cluster_songs.sample(min(n_recommendations, len(cluster_songs)))

        return recommendations[['title', 'artist', 'sentiment' , 'id']]

    def filter_songs_by_sentiment(recommended_songs, mood_sentiment, threshold=0.3):
        filtered_songs = []
        for _, row in recommended_songs.iterrows():
            song_sentiment = row["sentiment"]
            if abs(song_sentiment - mood_sentiment) <= threshold:
                filtered_songs.append(row)
        return pd.DataFrame(filtered_songs)




    model, scaler = train_model(df)


    # mood_sentiment = sentiment_score(user_mood)
    # recommended_songs = recommend_songs(df, model, scaler, user_mood)
    # filtered_songs = filter_songs_by_sentiment(recommended_songs, mood_sentiment)
    user_mood = input("Enter your mood: ").lower()

    if user_mood == 'sad' or user_mood == 'bad':
        sad_playlist_id = "4PWQV9dQpT7As9OTZBqrR8"
        sad_songs_df = create_dataset_from_playlist(sad_playlist_id)
        recommended_songs = recommend_songs(sad_songs_df, model, scaler, user_mood)
    else:
        recommended_songs = recommend_songs(df, model, scaler, user_mood)

    # recommended_songs = recommend_songs(df, model, scaler, user_mood)
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