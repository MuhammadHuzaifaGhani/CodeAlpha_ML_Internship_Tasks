import streamlit as st
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load trained model & encoder
# -----------------------------
model = load_model("emotion_model.h5")   # save your model after training as .h5
lb = joblib.load("label_encoder.pkl")    # save LabelEncoder with joblib

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_features(file_path, max_pad_len=174):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Pad/truncate for fixed size
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs

def predict_emotion(file_path):
    feature = extract_features(file_path)
    feature = feature.reshape(1, feature.shape[1], feature.shape[0])
    prediction = model.predict(feature)
    emotion = lb.inverse_transform([np.argmax(prediction)])
    return emotion[0]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a speech audio file (.wav) and the model will predict the emotion.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save temp file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict
    emotion = predict_emotion("temp.wav")
    st.success(f"Predicted Emotion: **{emotion}** 🎯")


