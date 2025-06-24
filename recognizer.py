# Ce fichier contient l'IA qui gère la reconnaissance vocale
# Si vous n'avez pas réentrainé le modele, les performances seront faibles, je vous recommande de changer l'import de recognizer.py dans le main.py par recognizer_google.py 

import librosa
import numpy as np
import tensorflow as tf
import pickle
import sounddevice as sd
from scipy.io.wavfile import write

class SpeechRecognizer:
    def __init__(self):
        # Charger le modèle et les mappages
        self.model = tf.keras.models.load_model("audio_ai/speech_to_text_model.keras")
        with open('audio_ai/char_to_idx.pkl', 'rb') as f:
            self.char_to_idx = pickle.load(f)
        with open('audio_ai/idx_to_char.pkl', 'rb') as f:
            self.idx_to_char = pickle.load(f)
        
        # Paramètres audio
        self.fs = 16000  # Fréquence d'échantillonnage
        self.duration = 5  # Durée d'enregistrement en secondes

    def record_audio(self):
        print("Enregistrement en cours... parlez maintenant")
        recording = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1)
        sd.wait()
        return recording.flatten(), self.fs

    def preprocess_audio(self, audio, sr):
        if sr != self.fs:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.fs)
        
        # Extraction des MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=self.fs, n_mfcc=50)
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        features = mfccs.T
        
        # Padding
        max_len = self.model.input_shape[1]
        if features.shape[0] < max_len:
            features = np.pad(features, ((0, max_len - features.shape[0]), (0,0)))
        else:
            features = features[:max_len, :]
        
        return np.expand_dims(features, axis=0)

    def predict_text(self, audio_input):
        predictions = self.model.predict(audio_input, verbose=0)[0]
        
        text = []
        for timestep in predictions:
            idx = np.argmax(timestep)
            if idx > 0:
                text.append(self.idx_to_char.get(idx, '?'))
        
        return ''.join(text)

    def recognize_speech(self):
        try:
            audio, sr = self.record_audio()
            processed_audio = self.preprocess_audio(audio, sr)

            # Prédiction
            text = self.predict_text(processed_audio)
            
            if text.strip():
                return text
            else:
                print("Je n'ai pas compris.")
                return ""
                
        except Exception as e:
            print(f"Erreur de reconnaissance : {e}")
            return ""

recognizer = SpeechRecognizer()

def recognize_speech():
    return recognizer.recognize_speech()