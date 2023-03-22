from flask import Flask, render_template, redirect, url_for, flash
from forms import MoodForm
from recommender import train_model, recommend_songs, play_song, sp, create_dataset
import pandas as pd
import os
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'defcf84275fabbea7c2bcf2c5ab175ce'

data_file = "data.pkl"

if os.path.exists(data_file):
    with open(data_file, 'rb') as f:
        df = pickle.load(f)
else:
    playlist_id = "37i9dQZF1DXdPec7aLTmlC"  # Replace with your own playlist ID
    df = create_dataset(playlist_id)
    with open(data_file, 'wb') as f:
        pickle.dump(df, f)

model, scaler = train_model(df)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = MoodForm()
    if form.validate_on_submit():
        user_mood = form.mood.data
        recommended_songs = recommend_songs(df, model, scaler, user_mood)
        return render_template('recommendations.html', recommended_songs=recommended_songs)
    return render_template('index.html', form=form)



if __name__ == '__main__':
    app.run(debug=True)

