from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('Bruno.F_fpl.csv')

@app.route('/')
def index():
    names = sorted(data['Name'].unique())
    clubs = sorted(data['VsClub'].unique())
    points = sorted(data['Points'].unique())
    seasons = sorted(data['Season'].unique())

    return render_template('index.html', names=names, clubs=clubs, points=points, seasons=seasons)

@app.route('/predict', methods=['POST'])
def predict():
    names = request.form.get('names')
    clubs = request.form.get('clubs')
    points = request.form.get('points')
    seasons = request.form.get('seasons')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[names, clubs, points, seasons]],
                               columns=['names', 'clubs', 'points', 'seasons'])

    print("Input Data:")
    print(input_data)