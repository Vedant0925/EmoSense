import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests

SPOTIPY_CLIENT_ID = "49f3c85f3e9747c088ff47c8f471ae9c"
SPOTIPY_CLIENT_SECRET = "9b1621df4a7b4a38a4b4ae3468ab3d40"
SPOTIPY_REDIRECT_URI = "your_redirect_uri"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope="user-top-read"))

def fetch_web_api(endpoint, method, token, body=None):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.request(method, f"https://api.spotify.com/{endpoint}", headers=headers, json=body)
    return response.json()

def get_top_tracks(token):
    endpoint = "v1/me/top/tracks?time_range=short_term&limit=5"
    return fetch_web_api(endpoint, "GET", token).get("items")

top_tracks = get_top_tracks(sp.auth_manager.get_access_token())

if top_tracks:
    for track in top_tracks:
        name = track["name"]
        artists = ", ".join([artist["name"] for artist in track["artists"]])
        print(f"{name} by {artists}")
