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

'''import os
from pathlib import Path
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
from flask import Flask, request, render_template, redirect, url_for

# --- App Setup ---
app = Flask(_name) # CORRECTED: _name to _name_
BASE_DIR = Path(_file).resolve().parent # CORRECTED: _file to _file_
UPLOAD_FOLDER = BASE_DIR / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

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
            # Ensure sufficient length for feature extraction, pad if necessary
            print("⚠ Audio too short to process for some features, padding with zeros.")
            y_trimmed = np.pad(y_trimmed, (0, max(0, 512 - len(y_trimmed)))) # Ensure non-negative padding


        # Placeholder: MFCCs (13), delta MFCCs (13), delta-delta MFCCs (13), ZCR mean (1), RMS mean (1) = 41 features
        mfcc_raw = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        delta_mfcc_raw = librosa.feature.delta(mfcc_raw)
        delta2_mfcc_raw = librosa.feature.delta(mfcc_raw, order=2)

        mfcc_feat = np.mean(mfcc_raw.T, axis=0)
        delta_mfcc_feat = np.mean(delta_mfcc_raw.T, axis=0)
        delta2_mfcc_feat = np.mean(delta2_mfcc_raw.T, axis=0)

        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_trimmed).T, axis=0, keepdims=True)
        rms_energy = np.mean(librosa.feature.rms(y=y_trimmed).T, axis=0, keepdims=True) # Renamed for clarity

        # Ensure all parts are 1D before hstack
        features = np.hstack((
            mfcc_feat.flatten(),
            delta_mfcc_feat.flatten(),
            delta2_mfcc_feat.flatten(),
            zcr.flatten(),
            rms_energy.flatten() # Use new name
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
        # IMPROVEMENT: Secure filename to prevent directory traversal or other attacks
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        if not filename: # If secure_filename returns empty (e.g., filename was just "..")
            return render_template("index.html", error_message="Invalid filename.")

        filepath = UPLOAD_FOLDER / filename
        try:
            file.save(filepath)

            raw_features = extract_features(filepath)

            if raw_features is None:
                return render_template("index.html", error_message="Could not extract features from the audio file. Please ensure it's a valid audio format and duration.")

            scaled_features = scaler.transform(raw_features.reshape(1, -1))
            probability = model.predict(scaled_features)[0][0]

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
            return render_template("index.html", error_message=f"An error occurred processing the file.")
        finally:
            if filepath.exists():
                os.remove(filepath)
    
    return redirect(request.url)

# --- Run App ---
if _name_ == "_main": # CORRECTED: _name and main to _name_ and "_main_"
    app.run(debug=True, host="0.0.0.0", port=5000) '''
import os
from pathlib import Path
import numpy as np
import librosa
import pywt
from scipy.stats import entropy
from pywt import WaveletPacket
from tensorflow.keras.models import load_model
import joblib
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import logging

logging.basicConfig(level=logging.ERROR)

# --- App Setup ---
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# --- Load Model and Scaler ---
#MODEL_PATH = BASE_DIR / "dnn_model" / "ASVSpoof_2021_add_2022_audio_model_v2.h5"
#SCALER_PATH = BASE_DIR / "dnn_model" / "scaler.pkl(1)"
MODEL_PATH   = BASE_DIR / "Model"/"ASVSpoof 2021+add_2022_audio_model_v2.h5"
SCALER_PATH = BASE_DIR / "Model"/"scaler.pkl"
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}", exc_info=True)
    model = None
    scaler = None

# Expected number of features
N_FEATURES = 41

# --- Feature Extraction ---
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
    return 1 - np.std(mfcc) / np.mean(mfcc) if np.mean(mfcc) != 0 else 0

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
        extracted_features = np.array([[spectral_feat, pitch_var] + wavelet_feats])
        if extracted_features.shape[1] != N_FEATURES:
            print(f"⚠️ Extracted {extracted_features.shape[1]} features, expected {N_FEATURES}.")
        return extracted_features
    except Exception as e:
        logging.error(f"❌ Feature extraction error: {e}", exc_info=True)
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
        filename = secure_filename(file.filename)
        if not filename:
            return render_template("index.html", error_message="Invalid filename.")
        unique_filename = generate_unique_filename(filename)
        filepath = UPLOAD_FOLDER / unique_filename
        try:
            file.save(filepath)
            features = extract_features(filepath)

            if features is None:
                return render_template("index.html", error_message="Could not extract features from the audio file. Please ensure it's a valid audio format and duration.")

            if features.shape[1] != N_FEATURES:
                return render_template("index.html", error_message=f"Extracted {features.shape[1]} features, but the model expects {N_FEATURES}. There's likely an issue with feature extraction.")

            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0][0]

            threshold = 0.5
            if prediction >= threshold:
                label = "Real Audio"
                confidence_score = prediction * 100
            else:
                label = "Fake Audio"
                confidence_score = (1 - prediction) * 100

            return render_template("index.html", prediction=label, confidence=f"{confidence_score:.2f}%")

        except Exception as e:
            logging.error(f"❌ An error occurred during prediction: {e}", exc_info=True)
            return render_template("index.html", error_message=f"An error occurred processing the file: {e}")
        finally:
            if filepath.exists():
                os.remove(filepath)

    return redirect(request.url)

def generate_unique_filename(filename):
    _, ext = os.path.splitext(filename)
    return f"{uuid.uuid4().hex}{ext}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
