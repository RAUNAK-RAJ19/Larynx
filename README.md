# 🎙️ Voice Spoof Detection System (Deep Learning)

An advanced **Voice Spoof Detection System** built using deep learning to classify audio into:

* ✅ **Real Voice (Bonafide)**
* 🤖 **Deepfake Voice (Logical Access - LA)**
* 🔊 **Replay Attack (Physical Access - PA)**

This project leverages **audio signal processing + CNN models** to detect spoofed audio using the **ASVspoof 2019 dataset**.

---

## 🚀 Features

* 🔥 Deep Learning model using CNN
* 🎧 Audio → Mel Spectrogram conversion
* 🧠 Multi-class classification (Real / Deepfake / Replay)
* ⚡ Scalable for large datasets (~30GB)
* 🎤 Real-time microphone detection
* 📂 File-based prediction
* 🌐 API backend for UI integration (FastAPI)

---

## 🧠 Tech Stack

* Python
* TensorFlow
* Librosa
* NumPy
* FastAPI (for backend)
* SoundDevice (real-time audio)

---

## 📁 Project Structure

```
voice_spoof_advanced/
│
├── train_all_in_one.py      # Training script (DL)
├── predict.py               # Predict audio file
├── realtime.py              # Real-time mic detection
├── app.py                   # Backend API
│
├── models/
│   └── spoof_model.h5       # Trained model
```

---

## 📊 Dataset

This project uses the **ASVspoof 2019 dataset**:

* **LA (Logical Access)** → AI-generated / deepfake voices
* **PA (Physical Access)** → replay attacks

### Dataset Structure

```
LA/
PA/
ASVspoof2019_LA_cm_protocols/
ASVspoof2019_PA_cm_protocols/
```

---

## ⚙️ Installation

```bash
pip install numpy librosa tensorflow sounddevice fastapi uvicorn
```

---

## 🏋️‍♂️ Training the Model

```bash
python train_all_in_one.py
```

* Loads LA + PA datasets
* Converts audio → spectrogram
* Trains CNN model
* Saves model in `/models`

---

## 🔍 Predict on Audio File

```bash
python predict.py
```

Output example:

```
Prediction: Deepfake
Confidence: 92.3%
```

---

## 🎤 Real-Time Detection

```bash
python realtime.py
```

* Records audio from microphone
* Predicts in real-time

---

## 🌐 Run Backend API (for UI)

```bash
uvicorn app:app --reload
```

Open API docs:

```
http://127.0.0.1:8000/docs
```

---

## 🔄 Workflow

```
Audio Input
   ↓
Mel Spectrogram (Librosa)
   ↓
CNN Model (TensorFlow)
   ↓
Prediction (Real / Deepfake / Replay)
```

---

## 📈 Model Details

* Input: Mel Spectrogram (128 features)
* Architecture:

  * Conv2D + MaxPooling layers
  * Global Average Pooling
  * Dense layers
* Output:

  * 3-class softmax classification

---

## 🧪 Evaluation

The model can be evaluated using:

* Accuracy
* Confusion Matrix
* Precision / Recall / F1-score

---

## 🚀 Future Improvements

* 🔥 Add attention-based models (CNN + Attention)
* 📊 Confusion matrix visualization
* 🌍 Deploy on cloud (AWS / Render)
* 🎨 Improve UI (drag & drop audio upload)
* ⚡ Optimize training using GPU

---

## 🏆 Project Highlights

* Handles **large-scale dataset (~30GB)**
* Uses **industry-level DL pipeline**
* Implements **real-time audio inference**
* Designed for **AI/ML portfolio & placements**

---

## 👨‍💻 Author

**Raunak Raj**

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to contribute!

---
