from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("wandering_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    speed = data["speed"]
    distance = data["distance"]
    time_outside = data["time_outside"]

    X = np.array([[speed, distance, time_outside]])
    prediction = model.predict(X)[0]

    result = "wandering" if prediction == 1 else "normal"
    return jsonify({"result": result})

@app.route("/")
def home():
    return "ML Wandering Detection API Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
