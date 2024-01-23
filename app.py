from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

df = pd.read_csv('House Price India.csv')
df1 = df.drop(['Date', 'id', 'Lattitude', 'Longitude', 'Postal Code'], axis=1)

df1 = df1.dropna()
numeric_features = df1.select_dtypes(include=['float64', 'int64']).columns
df1 = df1[numeric_features]

X = df1.drop('Price', axis=1)
y = df1['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form['feature_' + str(i)]) for i in range(1, 18)]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
