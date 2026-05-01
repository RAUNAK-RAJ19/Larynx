import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf


SAMPLE_RATE = 16000
DURATION = 10
SAMPLES = SAMPLE_RATE * DURATION
SPEC_WIDTH = 94
UNCERTAIN_MARGIN = 0.15

MODEL_CANDIDATES = (
	Path("spoof_model_finetuned.h5"),
	Path("spoof_model.h5"),
)


def resolve_model_path() -> Path:
	for candidate in MODEL_CANDIDATES:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(
		"Could not find spoof_model.h5 in the project root."
	)


def load_model():
	return tf.keras.models.load_model(resolve_model_path())


def record_audio(duration: int = DURATION, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
	print(f"Recording for {duration} seconds... speak now.")
	audio = sd.rec(
		int(duration * sample_rate),
		samplerate=sample_rate,
		channels=1,
		dtype="float32",
	)
	sd.wait()
	return audio.squeeze()


def save_wav(audio: np.ndarray, file_path: Path, sample_rate: int = SAMPLE_RATE) -> None:
	sf.write(file_path, audio, sample_rate)


def extract_spectrogram_from_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
	mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
	mel_db = librosa.power_to_db(mel).astype(np.float32)
	if mel_db.shape[1] < SPEC_WIDTH:
		mel_db = np.pad(mel_db, ((0, 0), (0, SPEC_WIDTH - mel_db.shape[1])))
	else:
		mel_db = mel_db[:, :SPEC_WIDTH]
	return mel_db


def predict_audio(model, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[str, float, float, float]:
	target_samples = sample_rate * 3
	# Pad if the entire audio is still less than 3 seconds
	if len(audio) < target_samples:
		audio = np.pad(audio, (0, target_samples - len(audio)))
		
	# 2. Slice the audio into 3-second overlapping chunks
	step = target_samples // 2  # 1.5 second overlap
	chunks = []
	for start_idx in range(0, max(1, len(audio) - target_samples + 1), step):
		chunk = audio[start_idx : start_idx + target_samples]
		if len(chunk) < target_samples:
			chunk = np.pad(chunk, (0, target_samples - len(chunk)))
		chunks.append(chunk)
		
	# 3. Extract spectrograms for all chunks
	spec_inputs = []
	for c in chunks:
		spec = extract_spectrogram_from_audio(c, sample_rate)
		spec_inputs.append(spec[..., np.newaxis])
		
	# 4. Predict on all chunks and calculate the average score
	prediction_batch = model.predict(np.array(spec_inputs), verbose=0)
	avg_pred = np.mean(prediction_batch, axis=0)

	prob_real = float(avg_pred[0])
	prob_ai = float(avg_pred[1])
	margin = abs(prob_real - prob_ai)

	if margin < UNCERTAIN_MARGIN:
		label = "UNCERTAIN"
		confidence = max(prob_real, prob_ai)
	elif prob_real >= prob_ai:
		label = "REAL (Human)"
		confidence = prob_real
	else:
		label = "AI / Deepfake"
		confidence = prob_ai

	return label, confidence, prob_real, prob_ai


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Record microphone audio and predict whether it is real or AI-generated."
	)
	parser.add_argument(
		"--output",
		default="myrec.wav",
		help="Path where the recorded audio will be saved.",
	)
	parser.add_argument(
		"--duration",
		type=int,
		default=DURATION,
		help="Recording duration in seconds.",
	)
	args = parser.parse_args()

	model = load_model()
	audio = record_audio(duration=args.duration)

	output_path = Path(args.output)
	save_wav(audio, output_path)
	audio, _ = librosa.load(output_path, sr=SAMPLE_RATE)

	label, confidence, prob_real, prob_ai = predict_audio(model, audio)

	print("\n" + "=" * 40)
	print(f"FILE: {output_path}")
	print(f"RESULT: {label}")
	print(f"CONFIDENCE: {confidence * 100:.2f}%")
	print(f"MARGIN: {abs(prob_real - prob_ai):.4f}")
	print("-" * 40)
	print(f"Real Score: {prob_real:.4f}")
	print(f"AI Score:   {prob_ai:.4f}")
	print("=" * 40)


if __name__ == "__main__":
	main()
