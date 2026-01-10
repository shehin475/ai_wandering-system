from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import json

app = Flask(__name__)

# ---------------- LOAD ML MODEL ----------------
model = joblib.load("wandering_model.pkl")

# ---------------- FIREBASE CONFIG ----------------
FIREBASE_DB_URL = "https://ai-wandering-system.firebaseio.com"
FCM_SERVER_KEY = "TPY2Zbk6sXpC3DjktvmuVV7U-S96BmsKdSTjCRW3h2U"

# ---------------- SEND PUSH FUNCTION ----------------
def send_push(token, title, body):
    url = "https://fcm.googleapis.com/fcm/send"

    headers = {
        "Authorization": f"key={FCM_SERVER_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "to": token,
        "notification": {
            "title": title,
            "body": body
        }
    }

    requests.post(url, headers=headers, data=json.dumps(payload))


# ---------------- ML PREDICTION API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    speed = data["speed"]
    distance = data["distance"]
    time_outside = data["time_outside"]
    patient_id = data["patientId"]   # ⬅ sent from Android

    X = np.array([[speed, distance, time_outside]])
    prediction = model.predict(X)[0]

    result = "wandering" if prediction == 1 else "normal"

    # 🚨 IF WANDERING → SEND PUSH
    if result == "wandering":
        # fetch patient data from Firebase
        firebase_url = f"{FIREBASE_DB_URL}/patients/{patient_id}.json"
        patient_data = requests.get(firebase_url).json()

        if patient_data and "fcmToken" in patient_data:
            token = patient_data["fcmToken"]

            send_push(
                token,
                "🚨 Wandering Alert",
                "Patient has moved outside the safe area"
            )

    return jsonify({"result": result})


@app.route("/")
def home():
    return "ML Wandering Detection API Running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
