Import necessary modules-lyricsgenius, spotipy, textblob, numpy, pickle, os, pygame

Use your spotify client id and client secret(Generated by creating an account/signing in to spotify developer page)

Use your Genius API key

For almost instant recommendations, download the two pickle files.(Shouldn't be a problem if you don't. The script is equipped to create a new one if an existing one isn't found)

**WHAT'S NEW?**

Songs are now recommended after a thorough sentimental analysis of the lyrics AND the audio features from Spotify.
The following audio features are used-['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'sentiment']
After recommending songs, they are also played using the pygame library.
After all are done playing, feedback is taken and f1-score is used as the main evaluation metric.