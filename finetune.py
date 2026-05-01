import os
import time
import random
import argparse
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf

SAMPLE_RATE = 16000
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION
SPEC_WIDTH = 94

NUM_CLIPS = 10
MODEL_PATH = "spoof_model.h5"
SAVE_PATH = "spoof_model_finetuned.h5"

BASE_DIR = r"c:\Users\RAUNAK\OneDrive\Desktop\MLDL flow\voice spoof\archive (1)"
LA_AUDIO = os.path.join(BASE_DIR, "LA", "LA", "ASVspoof2019_LA_train", "flac")
LA_LABEL = os.path.join(BASE_DIR, "LA", "LA", "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")

def process_audio_to_spec(audio, sample_rate=SAMPLE_RATE):
    target_samples = sample_rate * DURATION
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    else:
        audio = audio[:target_samples]

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    mel_db = librosa.power_to_db(mel).astype(np.float32)
    if mel_db.shape[1] < SPEC_WIDTH:
        mel_db = np.pad(mel_db, ((0, 0), (0, SPEC_WIDTH - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :SPEC_WIDTH]
    return mel_db

def get_ai_samples(count=10):
    print(f"Loading {count} AI samples from the dataset to retain knowledge...")
    ai_paths = []
    if not os.path.exists(LA_LABEL):
        print("Dataset label file not found! Please check paths.")
        return []
        
    with open(LA_LABEL, "r") as f:
        for line in f:
            parts = line.strip().split()
            file_id, label = parts[1], parts[-1]
            if label == "spoof":
                path = os.path.join(LA_AUDIO, file_id + ".flac")
                if os.path.exists(path):
                    ai_paths.append(path)
                    
    random.shuffle(ai_paths)
    selected = ai_paths[:count]
    
    specs = []
    for path in selected:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        specs.append(process_audio_to_spec(audio))
        
    return specs

def record_real_samples(count=10):
    print(f"\n--- MICROPHONE CALIBRATION ---")
    print(f"We need to record {count} short clips of your real voice.")
    print("Say something different each time (e.g., read a sentence off the screen).")
    specs = []
    for i in range(count):
        input(f"\nPress ENTER to start recording clip {i+1}/{count}...")
        print("Recording for 3 seconds... speak now!")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = audio.squeeze()
        specs.append(process_audio_to_spec(audio))
        print("Recorded!")
        time.sleep(0.5)
    return specs

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Base model {MODEL_PATH} not found!")
        return
        
    # 1. Gather Data (Class 0: Real, Class 1: AI)
    real_specs = record_real_samples(NUM_CLIPS)
    ai_specs = get_ai_samples(NUM_CLIPS)
    
    if not ai_specs:
        print("Could not load AI samples. Aborting.")
        return
        
    X = np.array(real_specs + ai_specs)
    X = X[..., np.newaxis] # Add channel dimension
    
    # Labels: 0 for Real (first NUM_CLIPS), 1 for AI (next NUM_CLIPS)
    y = np.array([0]*len(real_specs) + [1]*len(ai_specs))
    
    # Shuffle together
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # 2. Load Model
    print("\nLoading base model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Lower learning rate for fine-tuning
    # (Re-compile model so we don't drastically alter existing weights instantly)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Train
    print("\nStarting fine-tuning...")
    model.fit(X, y, epochs=10, batch_size=4, verbose=1)
    
    # 4. Save
    model.save(SAVE_PATH)
    print(f"\n✅ Fine-tuning complete. Model saved as {SAVE_PATH}")
    print("You can now run 'python realtime.py' and it will use this finetuned model!")

if __name__ == "__main__":
    main()