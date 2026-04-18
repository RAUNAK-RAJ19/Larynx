import os
import numpy as np
import librosa
import tensorflow as tf


MODEL_PATH = "spoof_model.h5" 
FILE_TO_TEST = "rec3.mp3"  

def predict_binary(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    
    SAMPLE_RATE = 16000
    DURATION = 3
    SAMPLES = SAMPLE_RATE * DURATION
    
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    mel_db = librosa.power_to_db(mel).astype(np.float32)
    
    
    spec_input = mel_db[np.newaxis, ..., np.newaxis]

    
    pred = model.predict(spec_input, verbose=0)[0]
    
    
    prob_real = pred[0]
    prob_deepfake = pred[1]

    
    if prob_real > prob_deepfake:
        result = "🟢 REAL (Human)"
        confidence = prob_real
    else:
        result = "🔴 DEEPFAKE (AI)"
        confidence = prob_deepfake

    
    print("\n" + "="*40)
    print(f"FILE: {os.path.basename(file_path)}")
    print(f"RESULT: {result}")
    print(f"CONFIDENCE: {confidence*100:.2f}%")
    print("-" * 40)
    print(f"Real Score: {prob_real:.4f}")
    print(f"AI Score:   {prob_deepfake:.4f}")
    print("="*40 + "\n")


predict_binary(FILE_TO_TEST)