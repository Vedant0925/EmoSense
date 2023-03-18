import lyricsgenius
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from textblob import TextBlob
import numpy as np


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
            data.append({**audio_features, "sentiment": sentiment, "title": title, "artist": artist})

    return pd.DataFrame(data)



playlist_id = "7j8yjWybvKkVu7d0SFyH2I"
df = create_dataset(playlist_id)


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


# def recommend_songs(df, model, scaler, mood, n_recommendations=10):
#     mood_sentiment = sentiment_score(mood)
#
#     mood_features = df.iloc[0][['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence']].values
#     mood_features = list(mood_features)
#     mood_features.append(mood_sentiment)
#     mood_features = np.array([mood_features])
#
#
#     mood_features_scaled = scaler.transform(mood_features)
#
#
#     mood_cluster = model.predict(mood_features_scaled)
#
#
#     recommendations = df[df['cluster'] == mood_cluster[0]].sample(n_recommendations)
#
#     return recommendations[['title', 'artist']]


def recommend_songs(df, model, scaler, mood, n_recommendations=10):
    mood_sentiment = sentiment_score(user_mood)


    numeric_features = df.select_dtypes(include=np.number).drop(columns='sentiment')


    mood_features = list(numeric_features.iloc[0].values)


    mood_features[-1] = mood_sentiment
    mood_features = np.array([mood_features])


    mood_features_scaled = scaler.transform(mood_features)


    mood_cluster = model.predict(mood_features_scaled)


    available_songs = df[df['cluster'] == mood_cluster[0]]


    n_songs = min(len(available_songs), n_recommendations)
    recommendations = available_songs.sample(n_songs)

    return recommendations[['title', 'artist']]


model, scaler = train_model(df)


user_mood = input("Enter your mood: ")


recommended_songs = recommend_songs(df, model, scaler, user_mood)
print("\nRecommended songs for your mood:")
print(recommended_songs)


