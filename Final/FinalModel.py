import os
import time
import pickle
import spotipy
import lyricsgenius
import pandas as pd
import numpy as np
import pygame
from spotipy.oauth2 import SpotifyOAuth
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


GENIUS_API_KEY = "DyZvAtC52zoMh8uy90ZHZ2RGnIaxpGqVLsMwAjwfWrP7UhkVEJFv-5NPcYn4UXHy"
client_id = "49f3c85f3e9747c088ff47c8f471ae9c"
client_secret = "9b1621df4a7b4a38a4b4ae3468ab3d40"
redirect_uri = "http://localhost:8000/callback"
scope = "user-library-read user-top-read user-read-recently-played"
genius = lyricsgenius.Genius(GENIUS_API_KEY, timeout=15)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope))

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

import time

def get_top_tracks(time_range='short_term', limit=100):
    tracks = []
    offset = 0
    while len(tracks) < limit:
        results = sp.current_user_top_tracks(limit=min(50, limit - len(tracks)), offset=offset, time_range=time_range)
        if not results['items']:
            print("No more items in top tracks.")
            break
        for item in results['items']:
            track_id = item['id']
            title = item['name']
            artist = item['artists'][0]['name']
            audio_features = sp.audio_features([track_id])[0]
            if not any([val is None for val in audio_features.values()]):  # Check if any audio feature is missing
                tracks.append((track_id, title, artist))
                print(f"Added track: {title} by {artist}")
            else:
                print(f"Skipped track due to missing audio features: {title} by {artist}")
        offset += 50
    return tracks





# Collaborative filtering
# def collaborative_filtering(tracks, n_neighbors=50):
#     track_ids = [track[0] for track in tracks]
#     audio_features = sp.audio_features(track_ids)
#     track_features = pd.DataFrame(audio_features)
#     scaler = StandardScaler()
#     track_features_scaled = scaler.fit_transform(track_features)
#     model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='euclidean')
#     model.fit(track_features_scaled)
#     _, indices = model.kneighbors(track_features_scaled)
#     return [tracks[idx] for idx in indices.flatten()]

def collaborative_filtering(tracks, n_neighbors=50):
    if not history:
        print("No tracks found in listening history.")
        return []
    track_ids = [track[0] for track in tracks]
    audio_features = sp.audio_features(track_ids)
    track_features = pd.DataFrame(audio_features).dropna()  # Add dropna() here
    scaler = StandardScaler()
    track_features_scaled = scaler.fit_transform(track_features)
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='euclidean')
    model.fit(track_features_scaled)
    _, indices = model.kneighbors(track_features_scaled)
    return [tracks[idx] for idx in indices.flatten()]
    if not track_features:
        print("No audio features found for the tracks in listening history.")
        return []


def sentiment_score(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
# Clustering and recommendations
def recommend_songs(tracks, model, scaler, mood, n_recommendations=10):
    mood_sentiment = sentiment_score(mood)
    track_features = [sp.audio_features(track[0])[0] for track in tracks]
    track_features_df = pd.DataFrame(track_features)
    track_features_df['sentiment'] = [mood_sentiment] * len(track_features_df)
    track_features_scaled = scaler.transform(track_features_df)
    mood_cluster = model.predict(track_features_scaled)
    cluster_songs = track_features_df[track_features_df['cluster'] == mood_cluster[0]]
    recommendations = cluster_songs.sample(min(n_recommendations, len(cluster_songs)))
    return recommendations.index


history = get_top_tracks(time_range='short_term', limit=100)



filtered_tracks = collaborative_filtering(history)


def create_dataset(tracks):
    data = []
    for track_id, title, artist in tracks:
        lyrics = get_lyrics(title, artist)
        if lyrics:
            audio_features = sp.audio_features([track_id])[0]
            sentiment = sentiment_score(lyrics)
            data.append({**audio_features, "sentiment": sentiment, "title": title, "artist": artist})
    return pd.DataFrame(data)

df = create_dataset(filtered_tracks)

def train_model(df, n_clusters=10):
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'sentiment']
    print("Dataframe columns:", df.columns)
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters)
    model.fit(X_scaled)

    df['cluster'] = model.labels_

    return model, scaler

model, scaler = train_model(df)


user_mood = input("Enter your mood: ")


recommended_indices = recommend_songs(filtered_tracks, model, scaler, user_mood)
recommended_songs = [filtered_tracks[idx] for idx in recommended_indices]


print("\nRecommended songs for your mood:")
for track_id, title, artist in recommended_songs:
    print(f"{title} by {artist}")


def play_song(spotify_uri):
    pygame.mixer.init()

    track = sp.track(spotify_uri)
    preview_url = track['preview_url']

    if preview_url is None:
        print("Sorry, no preview is available for this song.")
        return

    preview_file = f"{spotify_uri}_preview.mp3"
    if not os.path.exists(preview_file):
        import urllib.request
        urllib.request.urlretrieve(preview_url, preview_file)

    pygame.mixer.music.load(preview_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

for track_id, title, artist in recommended_songs:
    print(f"\nPlaying '{title}' by {artist}")
    track = sp.track(track_id)
    spotify_uri = track['uri']
    play_song(spotify_uri)
