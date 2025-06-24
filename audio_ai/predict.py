import os
import librosa
import numpy as np
import tensorflow as tf
import pickle

def load_assets():
    model = tf.keras.models.load_model("speech_to_text_model.keras")
    with open('char_to_idx.pkl', 'rb') as f:
        char_to_idx = pickle.load(f)
    with open('idx_to_char.pkl', 'rb') as f:
        idx_to_char = pickle.load(f)
    return model, char_to_idx, idx_to_char

def predict_audio(audio_path, model, idx_to_char):
    print(f"\nTraitement du fichier: {audio_path}")
    
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Durée audio: {len(audio)/sr:.2f} sec")
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=50)
    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
    features = mfccs.T
    
    max_len = model.input_shape[1]
    if features.shape[0] < max_len:
        features = np.pad(features, ((0, max_len - features.shape[0]), (0,0)))
    else:
        features = features[:max_len, :]
    
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features, verbose=0)[0]
    
    text = []
    for timestep in predictions:
        idx = np.argmax(timestep)
        if idx > 0:
            text.append(idx_to_char.get(idx, '?'))
    
    return ''.join(text)

model, char_to_idx, idx_to_char = load_assets()
print(f"Vocabulaire chargé ({len(char_to_idx)} caractères): {list(char_to_idx.keys())}")

# Test avec un fichier
test_file = "fichier_2.wav" # Fichier de test
if os.path.exists(test_file):
    predicted_text = predict_audio(test_file, model, idx_to_char)
    
    # Affiche la transcription réelle
    txt_file = test_file.replace("audio", "transcriptions").replace(".wav", ".txt")
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            true_text = f.read().strip()
        print(f"\nTexte réel: {true_text}")
    
    print(f"\nTexte prédit: {predicted_text}")
else:
    print("Fichier test non trouvé")