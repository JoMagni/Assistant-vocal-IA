import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt

AUDIO_DIR = "../data/audio"
TRANSCRIPTS_DIR = "../data/transcriptions"
MODEL_PATH = "speech_to_text_model.keras"
SAMPLE_RATE = 16000
DURATION = 20          # secondes
N_MFCC = 50            # nombre de features
MAX_TEXT_LENGTH = 200  # longueur maximale du texte

def load_data():
    audio_files = []
    texts = []
    
    for audio_file in os.listdir(AUDIO_DIR):
        if audio_file.endswith('.wav'):
            base_name = os.path.splitext(audio_file)[0]
            txt_file = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.txt")
            
            if os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                audio_path = os.path.join(AUDIO_DIR, audio_file)
                audio_files.append(audio_path)
                texts.append(text)
    
    return audio_files, texts

def extract_features(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < SAMPLE_RATE * DURATION:
            audio = np.pad(audio, (0, max(0, SAMPLE_RATE * DURATION - len(audio))))
        
        # Extraction des MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        return mfccs.T
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def prepare_text(texts):
    chars = set()
    for text in texts:
        chars.update(text.lower())

    char_to_idx = {c: i+1 for i, c in enumerate(sorted(chars))}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    encoded_texts = []
    for text in texts:
        encoded = [char_to_idx[c.lower()] for c in text if c.lower() in char_to_idx]
        encoded_texts.append(encoded)
    
    max_len = min(MAX_TEXT_LENGTH, max(len(t) for t in encoded_texts))
    padded_texts = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_texts, maxlen=max_len, padding='post', truncating='post'
    )
    
    return padded_texts, char_to_idx, idx_to_char, max_len

def prepare_datasets():
    audio_files, texts = load_data()
    print(f"Nombre total d'échantillons: {len(audio_files)}")
    
    features = []
    valid_texts = []
    for i, (audio_file, text) in enumerate(zip(audio_files, texts)):
        if i % 100 == 0:
            print(f"Traitement {i}/{len(audio_files)}...")
        
        feat = extract_features(audio_file)
        if feat is not None and text.strip():
            features.append(feat)
            valid_texts.append(text)
    
    padded_texts, char_to_idx, idx_to_char, max_text_len = prepare_text(valid_texts)
    num_chars = len(char_to_idx) + 1  # +1 pour le padding
    
    with open('char_to_idx.pkl', 'wb') as f:
        pickle.dump(char_to_idx, f)
    with open('idx_to_char.pkl', 'wb') as f:
        pickle.dump(idx_to_char, f)
    
    max_audio_len = max([f.shape[0] for f in features])
    features = np.array([np.pad(f, ((0, max_audio_len - f.shape[0]), (0,0))) for f in features])
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, padded_texts, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_chars, max_text_len, max_audio_len

def build_model(input_shape, num_chars, max_text_len):
    audio_input = tf.keras.layers.Input(shape=input_shape, name='audio_input')
    
    x = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(audio_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.RepeatVector(max_text_len)(x)
    
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(num_chars, activation='softmax')
    )(x)
    
    model = tf.keras.models.Model(inputs=audio_input, outputs=output)
    

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        run_eagerly=False
    )
    
    return model

def train():
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, num_chars, max_text_len, max_audio_len = prepare_datasets()
        
        input_shape = (max_audio_len, N_MFCC)
        model = build_model(input_shape, num_chars, max_text_len)
        
        print("\n=== Architecture du modèle ===")
        model.summary()
        
        print("\n=== Début de l'entraînement ===")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=32,
            epochs=100,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_accuracy', restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
            ],
            verbose=1
        )
        
        # Évaluation
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n=== Résultats finaux ===")
        print(f"Accuracy sur le test set: {test_acc:.2%}")
        
        # Sauvegarde
        model.save(MODEL_PATH)
        print(f"Modèle sauvegardé dans {MODEL_PATH}")
        
        return history
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
        return None

if __name__ == "__main__":
    train()