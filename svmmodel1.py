import streamlit as st
import os
import joblib
import librosa
import numpy as np

# Load the SVM model
model_path = 'svmmodel.pkl'
svm_classifier = joblib.load(model_path)

# Define the emotion label mapping
emotion_label_map = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fear",
    5: "disgust",
    6: "ps"
}

# Function to extract features and make predictions
def predict_emotion(audio_file):
    y, sr = librosa.load(audio_file)
    mfcc_features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma_features = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel_features = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

    # Combine the features into a single feature vector
    feature_vector = np.hstack((mfcc_features, chroma_features, mel_features))

    # Normalize the feature vector (using the same scaler as during training)
    feature_vector_normalized = scaler.transform(feature_vector.reshape(1, -1))

    # Predict the emotion label
    predicted_label = svm_classifier.predict(feature_vector_normalized)[0]

    return emotion_label_map[predicted_label]

# Streamlit UI
st.title('Emotion Identification from Audio')

# File uploader widget
uploaded_file = st.file_uploader('Upload an audio file (in WAV format):', type=['wav'])

if uploaded_file is not None:
    # Display the uploaded audio file
    st.audio(uploaded_file, format='audio/wav')

    # Make a prediction when the user clicks the button
    if st.button('Predict Emotion'):
        # Save the uploaded audio file to a temporary location
        with open('temp.wav', 'wb') as f:
            f.write(uploaded_file.read())

        # Predict the emotion
        predicted_emotion = predict_emotion('temp.wav')

        # Display the predicted emotion
        st.write(f'Predicted Emotion: {predicted_emotion}')
