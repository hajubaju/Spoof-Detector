import os
from pathlib import Path
import numpy as np
import librosa
import pywt
from scipy.stats import entropy
from pywt import WaveletPacket
import joblib
from flask import Flask, request, render_template
from keras.models import Sequential


# ─── APP SETUP ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR     = Path(__file__).parent.resolve()
UPLOADS_DIR  = BASE_DIR / "uploads"
MODEL_PATH   = BASE_DIR / "dnn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# make sure uploads folder exists
os.makedirs(UPLOADS_DIR, exist_ok=True)
    # Load model data
model_dict = joblib.load("dnn_model.pkl")

# Rebuild model from config and set weights
model = Sequential.from_config(model_dict["config"])
weights = [np.array(w) for w in model_dict["weights"]]
model.set_weights(weights)
print("Model loaded successfully!")

# ─── FEATURE EXTRACTION ─────────────────────────────────────────────────────────
def load_audio(file_path):
    y, sr = librosa.load(str(file_path), sr=16000)
    y = y / np.max(np.abs(y))
    return sr, y

def ewt_decompose(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    return coeffs

def wpt_decompose(signal, wavelet='db4', level=3):
    wp = WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
    nodes = [node.path for node in wp.get_level(level)]
    return [wp[node].data for node in nodes]

def extract_wavelet_features(ewt_coeffs, wpt_coeffs):
    features = []
    for coeff in ewt_coeffs:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    for coeff in wpt_coeffs:
        features += [np.mean(coeff), np.var(coeff), entropy(np.abs(coeff) + 1e-10)]
    return features

def extract_spectral_feature(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return 1 - np.std(mfcc) / np.mean(mfcc)

def extract_pitch_variation(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid = pitches[magnitudes > np.median(magnitudes)]
    return np.std(valid) / np.mean(valid) if len(valid) > 0 and np.mean(valid) > 0 else 0

def extract_features(file_path):
    try:
        sr, y = load_audio(file_path)
        ewt_coeffs = ewt_decompose(y)
        wpt_coeffs = wpt_decompose(y)
        wavelet_feats = extract_wavelet_features(ewt_coeffs, wpt_coeffs)
        spectral_feat = extract_spectral_feature(y, sr)
        pitch_var = extract_pitch_variation(y, sr)
        return np.array([[spectral_feat, pitch_var] + wavelet_feats])
    except Exception as e:
        print("❌ Feature extraction error:", e)
        return None


# ─── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    # show upload/record page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # get the uploaded file
    audio_file = request.files.get("audio_file")
    if not audio_file or audio_file.filename == "":
        return render_template("index.html", message="No file selected")
    if not audio_file.filename.lower().endswith((".wav", ".webm", ".flac")):
        return render_template("index.html", message="Only .wav, .webm or flac allowed")

    # save temporarily
    save_path = UPLOADS_DIR / audio_file.filename
    audio_file.save(save_path)

    # extract features & remove temp file
    features = extract_features(save_path)
    save_path.unlink()

    if features is None:
        return render_template("index.html", message="Failed to extract features")
    # Normalize features
    features_scaled = scaler.transform(features)

    # predict
    score = float(model.predict(features)[0][0])
    label = "Genuine" if score < 0.5 else "Deepfake"

    # render the same template, passing in result
    return render_template("index.html",
                           label=label,
                           confidence=round(score * 100, 2))

# ─── RUN ────────────────────────────────────────────────────────────────────────
if __name__== "_main_":
    app.run(host="0.0.0.0", port=5000, debug=True)
