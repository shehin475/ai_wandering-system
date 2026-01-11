from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import json

from google.oauth2 import service_account
from google.auth.transport.requests import Request

from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)

# ---------------- LOAD ML MODEL ----------------
model = joblib.load("wandering_model.pkl")

# ---------------- FIREBASE CONFIG ----------------
FIREBASE_DB_URL = "https://ai-wandering-system.firebaseio.com"
import os

SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")


# ---------------- FCM HTTP v1 TOKEN ----------------
def get_access_token():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/firebase.messaging"]
    )
    credentials.refresh(Request())
    return credentials.token

# ---------------- SEND PUSH (HTTP v1) ----------------
def send_push(token, title, body):
    access_token = get_access_token()

    url = f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "message": {
            "token": token,
            "notification": {
                "title": title,
                "body": body
            },
            "android": {
                "priority": "HIGH"
            }
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    print("FCM STATUS:", response.status_code)
    print("FCM RESPONSE:", response.text)

# ---------------- ML PREDICTION API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    speed = data["speed"]
    distance = data["distance"]
    time_outside = data["time_outside"]
    patient_id = data["patientId"]

    X = np.array([[speed, distance, time_outside]])
    prediction = model.predict(X)[0]

    result = "wandering" if prediction == 1 else "normal"

    # 🚨 IF WANDERING → SEND PUSH
    if result == "wandering":
        firebase_url = f"{FIREBASE_DB_URL}/patients/{patient_id}.json"
        patient_data = requests.get(firebase_url).json()

        if patient_data and "fcmToken" in patient_data:
            send_push(
                patient_data["fcmToken"],
                "🚨 Wandering Alert",
                "You are outside the safe area. Please return."
            )

    return jsonify({"result": result})

# ---------------- HEALTH CHECK ----------------
@app.route("/")
def home():
    return "AI Wandering Detection ML API Running (FCM HTTP v1)"

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
