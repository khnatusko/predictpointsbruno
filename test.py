from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

app = Flask(__name__)

# Zakładamy, że model trenowany jest w trakcie uruchomienia aplikacji
data = pd.read_csv('Bruno.F_fpl.csv')
#Usunięcie niepotrzebnych kolumn i obsługa brakujących wartości
data = data.dropna()
#Definiowanie cech (features) i etykiet (label)
X = data[['minutes_played', 'goals_scored', 'assists', 'clean_sheets', 'yellow_cards', 'red_cards', 'opponent_difficulty']]
y = data['total_points']

#zakodowanie zmiennej player, jako zmienna kategoryczna i zmienienie jej wartości na wartość numeryczną
if 'player' in data.columns:
    position_dummies = pd.get_dummies(data['player'], prefix='player')
    X = pd.concat([X, position_dummies], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu regresji
model = LinearRegression()
model.fit(X_train, y_train)

# Strona główna z formularzem
@app.route('/')
def index():
    return render_template('./index.html')

y_pred = model.predict(X_test)

# Ocena modelu
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Średni błąd bezwzględny (MAE): {mae}")
print(f"Pierwiastek średniego błędu kwadratowego (RMSE): {rmse}")

# Krok 5: Predykcja punktów dla konkretnego zawodnika (przykład)
# Zawodnik, który zagrał 90 minut, strzelił 1 bramkę, 1 asystę, bez żółtych i czerwonych kartek, przeciwnik o trudności 2
new_player_data = [[X_train]]

predicted_points = model.predict(new_player_data)
print(f"Przewidywane punkty zawodnika: {predicted_points}")

if __name__ == '__main__':
    app.run(debug=True)