import lyricsgenius
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from textblob import TextBlob
import numpy as np
import pickle
import os
import pygame
import threading
from flask import Flask, request, jsonify



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


# user_mood = input("Enter your mood: ")

# if user_mood.lower() == "sad" or user_mood.lower()=="bad":
#     playlist_id = "4PWQV9dQpT7As9OTZBqrR8"
# else:
#     playlist_id = "37i9dQZF1DXdPec7aLTmlC"
# if language.lower()=='english':
#     if user_mood.lower() == "sad" or user_mood.lower() == "bad":
#         playlist_id = "4PWQV9dQpT7As9OTZBqrR8"
#     else:
#         playlist_id = "37i9dQZF1DXdPec7aLTmlC"


# data_file = f"{playlist_id}_data.pkl"
#
# if os.path.exists(data_file):
#     with open(data_file, 'rb') as f:
#         df = pickle.load(f)
# else:
#     df = create_dataset(playlist_id)
#     with open(data_file, 'wb') as f:
#         pickle.dump(df, f)



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

    print(f"Mood sentiment: {mood_sentiment}")  # Add this line for debugging

    mood_features = df.iloc[0][
        ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness',
         'valence']].values
    mood_features = list(mood_features)
    mood_features.append(mood_sentiment)
    mood_features = np.array([mood_features])

    mood_features_scaled = scaler.transform(mood_features)

    mood_cluster = model.predict(mood_features_scaled)

    print(f"Mood cluster: {mood_cluster}")  # Add this line for debugging

    cluster_songs = df[df['cluster'] == mood_cluster[0]]
    recommendations = cluster_songs.sample(min(n_recommendations, len(cluster_songs)))

    return recommendations[['title', 'artist']]


# model, scaler = train_model(df)

# recommended_songs = recommend_songs(df, model, scaler, user_mood)
# print("\nRecommended songs for your mood:")
# print(recommended_songs)

import pygame


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

# for _, row in recommended_songs.iterrows():
#     print(f"\nPlaying '{row['title']}' by {row['artist']}")
#     track = sp.search(f"{row['title']} {row['artist']}", type='track', limit=1)
#     if track['tracks']['items']:
#         spotify_uri = track['tracks']['items'][0]['uri']
#         play_song(spotify_uri)
#     else:
#         print(f"Couldn't find the song '{row['title']}' by {row['artist']}' on Spotify.")


def get_user_feedback(recommended_songs):
    relevant_count = 0
    total_recommended = len(recommended_songs)
    for _, row in recommended_songs.iterrows():
        print(f"Did you like '{row['title']}' by {row['artist']}? (yes/no)")
        user_feedback = input().lower()
        if user_feedback == 'yes':
            relevant_count += 1
    return relevant_count, total_recommended

def evaluate_model(relevant_count, total_recommended, total_relevant):
    precision = relevant_count / total_recommended if total_recommended > 0 else 0
    recall = relevant_count / total_relevant if total_relevant > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


# relevant_count, total_recommended = get_user_feedback(recommended_songs)
#
#
# total_relevant = 10
#
# precision, recall, f1_score = evaluate_model(relevant_count, total_recommended, total_relevant)
#
# # print(f"Precision: {precision:.2f}")
# # print(f"Recall: {recall:.2f}")
# print(f"F1-score: {f1_score:.2f}")
#
# if 0<=f1_score<=0.4:
#
#     print("Apologies for such recommendations. We shall try to do better next time")
#
#
# elif 0.4<=f1_score<=0.6:
#
#     print("Just okay then? We appreciate it but should certainly try to improve")
#
#
# else:
#
#     print("Great! We're glad you love our recommendations")


