"""import os
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
if __name__== "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)"""


import os
from pathlib import Path
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
from flask import Flask, request, render_template, redirect, url_for

# --- App Setup ---
app = Flask(_name_)
BASE_DIR = Path(_file_).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER) # Flask config needs string path
'''app = Flask(__name__)

BASE_DIR     = Path(__file__).parent.resolve()
UPLOADS_DIR  = BASE_DIR / "uploads"
MODEL_PATH   = BASE_DIR / "dnn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl" '''
# --- Load Model and Scaler ---
MODEL_PATH = BASE_DIR / "model" / "ASVSpoof_2021_add_2022_audio_model_v2.h5"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    model = None
    scaler = None

# Expected number of features from your notebook (derived from model summary)
N_FEATURES = 41

# --- Feature Extraction ---
def extract_features(audio_path):
    """
    Extracts features from an audio file.
    🚨 IMPORTANT: This function MUST be modified to extract the exact same
    41 features in the exact same order as used to train your model
    (i.e., the features present in your CSV files).
    The current implementation is a placeholder using common audio features.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=16000) # Use consistent sampling rate

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) < 512: # Check if audio is too short after trimming
             print("⚠ Audio too short after trimming, using original audio.")
             y_trimmed = y
        if len(y_trimmed) < 512: # Still too short
             print("⚠ Audio too short to process for some features.")
             # Pad with zeros to allow librosa functions to work
             y_trimmed = np.pad(y_trimmed, (0, 512 - len(y_trimmed)))


        # Placeholder: MFCCs (13), delta MFCCs (13), delta-delta MFCCs (13), ZCR mean (1), RMS mean (1) = 41 features
        # This is a common set but MIGHT NOT MATCH YOUR CSV FEATURES.
        mfcc_raw = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        delta_mfcc_raw = librosa.feature.delta(mfcc_raw)
        delta2_mfcc_raw = librosa.feature.delta(mfcc_raw, order=2)

        mfcc_feat = np.mean(mfcc_raw.T, axis=0)
        delta_mfcc_feat = np.mean(delta_mfcc_raw.T, axis=0)
        delta2_mfcc_feat = np.mean(delta2_mfcc_raw.T, axis=0)

        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_trimmed).T, axis=0, keepdims=True)
        rms = np.mean(librosa.feature.rms(y=y_trimmed).T, axis=0, keepdims=True)

        # Ensure all parts are 1D before hstack
        features = np.hstack((
            mfcc_feat.flatten(),
            delta_mfcc_feat.flatten(),
            delta2_mfcc_feat.flatten(),
            zcr.flatten(),
            rms.flatten()
        ))

        if features.shape[0] != N_FEATURES:
            print(f"❌ Feature extraction error: Expected {N_FEATURES} features, but got {features.shape[0]}.")
            print("🚨 YOU MUST UPDATE THE extract_features FUNCTION in app.py to match your training data features.")
            return None
        
        return features

    except Exception as e:
        print(f"❌ Error extracting features from {audio_path}: {e}")
        return None

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, confidence=None, error_message=None)

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return render_template("index.html", error_message="Model or scaler not loaded. Check server logs.")

    if 'audio_file' not in request.files:
        return redirect(request.url)
    
    file = request.files['audio_file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = file.filename # You might want to secure the filename
        filepath = UPLOAD_FOLDER / filename
        try:
            file.save(filepath)

            # Extract features
            raw_features = extract_features(filepath)

            if raw_features is None:
                return render_template("index.html", error_message="Could not extract features from the audio file. Please ensure it's a valid audio format and duration.")

            # Scale features
            # The scaler expects a 2D array, so reshape the 1D features
            scaled_features = scaler.transform(raw_features.reshape(1, -1))

            # Make prediction
            probability = model.predict(scaled_features)[0][0] # Model output is a probability

            # Determine label and confidence
            threshold = 0.5
            if probability >= threshold:
                label = "Real Audio"
                confidence_score = probability * 100
            else:
                label = "Fake Audio"
                confidence_score = (1 - probability) * 100

            return render_template("index.html", prediction=label, confidence=f"{confidence_score:.2f}%")

        except Exception as e:
            print(f"❌ An error occurred during prediction: {e}")
            return render_template("index.html", error_message=f"An error occurred: {e}")
        finally:
            # Clean up uploaded file
            if filepath.exists():
                os.remove(filepath)
    
    return redirect(request.url)

# --- Run App ---
if _name_ == "_main_":
    app.run(debug=True, host="0.0.0.0", port=5000)
