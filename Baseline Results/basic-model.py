
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


spotify_data = pd.read_csv('spotify_data.csv')
lyrics_data = pd.read_csv('lyrics_data.csv')


merged_data = pd.merge(spotify_data, lyrics_data, on='song_id')


vectorizer = TfidfVectorizer()
lyrics_matrix = vectorizer.fit_transform(merged_data['lyrics'])


lyrics_similarity = cosine_similarity(lyrics_matrix)


def recommend_songs(song_id, num_recommendations=10):
    
    song_index = merged_data[merged_data['song_id'] == song_id].index[0]
    
    song_similarities = lyrics_similarity[song_index]
    
    similar_song_indices = np.argsort(-song_similarities)[1:num_recommendations+1]
    
    return list(merged_data.iloc[similar_song_indices]['song_id'])

# example usage: recommend 10 songs similar to "7 rings" by Ariana Grande
recommendations = recommend_songs('6ocbgoVGwYJhOv1GgI9NsF', num_recommendations=10)
print(recommendations)
