# import required libraries
import pandas as pd
import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


spotify_data = pd.read_csv('spotify_data.csv')
lyrics_data = pd.read_csv('lyrics_data.csv')


merged_data = pd.merge(spotify_data, lyrics_data, on='song_id')


vectorizer = TfidfVectorizer()
lyrics_matrix = vectorizer.fit_transform(merged_data['lyrics'])


lyrics_similarity = cosine_similarity(lyrics_matrix)


def recommend_songs(song_id, mood, num_recommendations=10):

    song_index = merged_data[merged_data['song_id'] == song_id].index[0]

    song_similarities = lyrics_similarity[song_index]

    mood_data = merged_data[merged_data['mood'] == mood]

    mood_indices = mood_data.index

    mood_similarities = song_similarities[mood_indices]

    similar_song_indices = mood_indices[np.argsort(-mood_similarities)][1:num_recommendations+1]

    return list(merged_data.iloc[similar_song_indices]['song_id'])

# example usage
song_id = input("Enter song id: ")
mood = input("Enter mood: ")
num_recommendations = int(input("Number of songs you'd like: "))
recommended_songs = recommend_songs(song_id, mood, num_recommendations)
print(recommended_songs)
