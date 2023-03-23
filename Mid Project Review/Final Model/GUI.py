import tkinter as tk
from tkinter import messagebox
from MidReviewModel import recommend_songs, play_song, get_lyrics, sentiment_score, train_model, create_dataset, sp
import pickle
import os

GENIUS_API_KEY = "DyZvAtC52zoMh8uy90ZHZ2RGnIaxpGqVLsMwAjwfWrP7UhkVEJFv-5NPcYn4UXHy"
SPOTIPY_CLIENT_ID = "aa087ea1788347d1a6a4b30cbd6fdd34"
SPOTIPY_CLIENT_SECRET = "cac4f4e8e70f4a35b163665d533dd479"

user_mood = None
playlist_id = None

def on_recommend_click():
    global user_mood, playlist_id
    user_mood = entry_mood.get()
    if not user_mood:
        messagebox.showwarning("Warning", "Please enter your mood.")
        return

    if user_mood.lower() == "sad" or user_mood.lower() == "bad":
        playlist_id = "4PWQV9dQpT7As9OTZBqrR8"
    else:
        playlist_id = "37i9dQZF1DXdPec7aLTmlC"

    data_file = f"{playlist_id}_data.pkl"

    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            df = pickle.load(f)
    else:
        df = create_dataset(playlist_id)
        with open(data_file, 'wb') as f:
            pickle.dump(df, f)

    model, scaler = train_model(df)
    recommended_songs = recommend_songs(df, model, scaler, user_mood)
    update_song_list(recommended_songs)

def update_song_list(songs):
    listbox_songs.delete(0, tk.END)
    for _, row in songs.iterrows():
        listbox_songs.insert(tk.END, f"{row['title']} by {row['artist']}")

def on_play_click():
    selected_song = listbox_songs.get(listbox_songs.curselection())
    title, artist = selected_song.split(" by ")
    track = sp.search(f"{title} {artist}", type='track', limit=1)
    if track['tracks']['items']:
        spotify_uri = track['tracks']['items'][0]['uri']
        play_song(spotify_uri)
    else:
        messagebox.showerror("Error", f"Couldn't find the song '{title}' by {artist}' on Spotify.")

# Create a basic tkinter window
root = tk.Tk()
root.title("Mood-based Song Recommender")

# Add widgets
label_mood = tk.Label(root, text="Enter your mood:")
entry_mood = tk.Entry(root)
button_recommend = tk.Button(root, text="Recommend Songs", command=on_recommend_click)

listbox_songs = tk.Listbox(root)
button_play = tk.Button(root, text="Play", command=on_play_click)

# Place widgets
label_mood.grid(row=0, column=0)
entry_mood.grid(row=0, column=1)
button_recommend.grid(row=0, column=2)

listbox_songs.grid(row=1, column=0, columnspan=3)
button_play.grid(row=2, column=1)

root.mainloop()
