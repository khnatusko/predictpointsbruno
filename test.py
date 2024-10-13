from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Zakładamy, że model trenowany jest w trakcie uruchomienia aplikacji
data = pd.read_csv('fpl_player_data.csv')
X = data[['minutes_played', 'goals_scored', 'assists', 'clean_sheets', 'yellow_cards', 'red_cards', 'opponent_difficulty']]
y = data['total_points']

# Trening modelu regresji
model = LinearRegression()
model.fit(X, y)

# Strona główna z formularzem
@app.route('/')
def index():
    return render_template('index.html')

# Obsługa predykcji po wysłaniu formularza
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Pobieranie danych z formularza
        minutes_played = float(request.form['minutes_played'])
        goals_scored = float(request.form['goals_scored'])
        assists = float(request.form['assists'])
        clean_sheets = float(request.form['clean_sheets'])
        yellow_cards = float(request.form['yellow_cards'])
        red_cards = float(request.form['red_cards'])
        opponent_difficulty = float(request.form['opponent_difficulty'])

        # Przewidywanie punktów
        new_player_data = [[minutes_played, goals_scored, assists, clean_sheets, yellow_cards, red_cards, opponent_difficulty]]
        predicted_points = model.predict(new_player_data)[0]

        return render_template('index2.html', prediction=round(predicted_points, 2))

    except Exception as e:
        return render_template('index2.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)