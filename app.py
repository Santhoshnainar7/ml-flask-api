
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("house_model.pkl")

# Root endpoint
@app.route("/")
def home():
    return "House Price Prediction API is Running!"

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [[
        data["area"],
        data["bedrooms"],
        data["bathrooms"]
    ]]

    prediction = model.predict(features)

    return jsonify({
        "prediction": round(float(prediction[0]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
