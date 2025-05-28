# app.py

import os
from pathlib import Path

import numpy as np
import librosa
import pickle
from flask import Flask, request, render_template
from keras.models import Sequential
import joblib

# Load model data
model_dict = joblib.load("dnn_model.pkl")

# Rebuild model from config and set weights
model = Sequential.from_config(model_dict["config"])
model.set_weights(model_dict["weights"])


# ─── APP SETUP ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR     = Path(__file__).parent.resolve()
UPLOADS_DIR  = BASE_DIR / "uploads"
MODEL_PATH   = BASE_DIR / "dnn_model.pkl"

# make sure uploads folder exists
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ─── LOAD PICKLED MODEL ─────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} …")
with open(MODEL_PATH, "rb") as f:
    model_dict = pickle.load(f)

# rebuild the Keras model from the saved JSON + weights
model_dict = joblib.load("dnn_model.pkl")
model = Sequential.from_config(model_dict["config"])
model.set_weights(model_dict["weights"])
print("Model loaded successfully!")

# ─── FEATURE EXTRACTION ─────────────────────────────────────────────────────────
def extract_features(file_path, n_mfcc=41):
    try:
        audio, sr  = librosa.load(str(file_path), sr=None)
        mfccs      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        if mfccs_mean.shape[0] < n_mfcc:
            mfccs_mean = np.pad(mfccs_mean, (0, n_mfcc - mfccs_mean.shape[0]))
        return mfccs_mean.reshape(1, -1)
    except Exception as e:
        print("Feature extraction error:", e)
        return None

# ─── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    audio_file = request.files.get("audio_file")
    if not audio_file or audio_file.filename == "":
        return render_template("index.html", message="No file selected")
    if not audio_file.filename.lower().endswith((".wav", ".webm")):
        return render_template("index.html", message="Only .wav or .webm allowed")

    save_path = UPLOADS_DIR / audio_file.filename
    audio_file.save(save_path)

    features = extract_features(save_path)
    save_path.unlink()

    if features is None:
        return render_template("index.html", message="Failed to extract features")

    score = float(model.predict(features)[0][0])
    label = "Genuine" if score < 0.5 else "Deepfake"
    return render_template("result.html", label=label, confidence=round(score * 100, 2))

if __name__ == "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)