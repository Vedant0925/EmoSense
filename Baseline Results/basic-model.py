import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SPOTIFY_CLIENT_ID = 'aa087ea1788347d1a6a4b30cbd6fdd34'
SPOTIFY_CLIENT_SECRET = 'cac4f4e8e70f4a35b163665d533dd479'

spotify_credentials = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=spotify_credentials)

genius = lyricsgenius.Genius('DyZvAtC52zoMh8uy90ZHZ2RGnIaxpGqVLsMwAjwfWrP7UhkVEJFv-5NPcYn4UXHy')

lyrics_df = pd.read_csv('/Users/vedant/Desktop/IR_Datasets/lyrics.csv', usecols=['artist', 'song_name', 'lyrics'])

analyzer = SentimentIntensityAnalyzer()


lyrics_df['sentiment'] = lyrics_df['lyrics'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

user_mood = input("What is your current mood? ")


if user_mood in ['happy', 'cheerful', 'upbeat']:
    query = 'mood:happy OR mood:cheerful OR mood:upbeat'
    
elif user_mood in ['sad', 'melancholy', 'depressed']:
    query = 'mood:sad OR mood:melancholy OR mood:depressed'
    
elif user_mood in ['angry', 'frustrated', 'irritated']:
    query = 'mood:angry OR mood:frustrated OR mood:irritated'
    
else:
    query = 'mood:' + user_mood

results = spotify.search(q=query, type='track', limit=10)
tracks = results['tracks']['items']

sentiments = []

for track in tracks:
    song = genius.search_song(track['name'], track['artists'][0]['name'])
    if song:
        lyrics = song.lyrics
        sentiment = analyzer.polarity_scores(lyrics)['compound']
        sentiments.append(sentiment)

mean_sentiment = sum(sentiments) / len(sentiments)

if mean_sentiment > 0.5:
    print("You seem to be in a good mood. Here are some upbeat songs you might like:")
    upbeat_songs = lyrics_df[lyrics_df['lyrics'].str.contains(user_mood, case=False) & (lyrics_df['sentiment'] > 0.5)].sample(10)['song_name'].tolist()
    for song in upbeat_songs:
        print(song)

elif mean_sentiment < -0.5:
    print("You seem to be in a bad mood. Here are some calming songs you might like:")
    calming_songs = lyrics_df[lyrics_df['lyrics'].str.contains(user_mood, case=False) & (lyrics_df['sentiment'] < -0.5)].sample(10)['song_name'].tolist()
    for song in calming_songs:
        print(song)

else:
    print("You seem to be in a neutral mood. Here are some popular songs you might like:")
    popular_songs = lyrics_df[lyrics_df['lyrics'].str.contains(user_mood, case=False) | lyrics_df['song_name'].str.contains(user_mood,case=False)].sample(10)['song_name'].tolist()

    for song in popular_songs:
        print(song)

