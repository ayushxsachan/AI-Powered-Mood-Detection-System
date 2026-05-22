# AI-Powered Mood Detection System

An intelligent **AI-based Mood Detection System** that analyzes facial expressions in real time using a webcam and predicts human emotions such as happiness, sadness, anger, surprise, and neutrality.

The project combines **Computer Vision**, **Deep Learning**, and **Facial Emotion Recognition** to create an interactive human emotion analysis system.

---

## 🚀 Features

- 🎥 Real-time facial emotion detection
- 😊 Detects multiple emotions accurately
- 🧠 AI-powered prediction using Deep Learning
- 📷 Webcam-based live analysis
- ⚡ Fast and lightweight system
- 📊 Displays emotion confidence scores
- 🖥️ User-friendly interface

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – Face detection & image processing
- **TensorFlow / Keras** – Deep learning model
- **NumPy** – Numerical computations
- **MediaPipe / Haar Cascade** – Face landmark detection
- **Matplotlib** – Visualization (optional)

---

## 📂 Project Structure

```bash
AI-Mood-Detection-System/
│
├── main.py                 # Main application
├── model.h5                # Trained emotion detection model
├── emotion_labels.py       # Emotion class labels
├── requirements.txt        # Required libraries
├── README.md               # Project documentation
└── assets/                 # Demo images/screenshots
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai-mood-detection-system.git
cd ai-mood-detection-system
```

---

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install opencv-python tensorflow keras numpy matplotlib
```

---

## ▶️ Usage

Run the project using:

```bash
python main.py
```

Requirements:
- Webcam access enabled
- Good lighting conditions
- Face clearly visible in frame

---

## 😊 Detected Emotions

| Emotion | Description |
|----------|-------------|
| Happy | Positive facial expression |
| Sad | Low emotional state |
| Angry | Aggressive facial cues |
| Surprise | Shocked expression |
| Neutral | Emotionally balanced |
| Fear | Anxiety/fear indicators |
| Disgust | Negative reaction |

---

## 🧠 How It Works

1. Webcam captures live video feed.
2. OpenCV detects faces in each frame.
3. Detected face is preprocessed and resized.
4. Deep Learning model predicts emotion.
5. Predicted mood is displayed in real time with confidence score.

---

## 🤖 AI Model

The system uses a **Convolutional Neural Network (CNN)** trained on facial emotion datasets such as:

- FER-2013
- CK+
- Custom emotion datasets

### Model Workflow

- Face Detection
- Image Preprocessing
- Feature Extraction
- Emotion Classification
- Real-Time Prediction

---

## 📸 Demo

Add screenshots or GIFs here.

Example:

```bash
assets/demo.gif
```

---

## 🔥 Challenges Faced

- Handling low-light conditions
- Improving emotion prediction accuracy
- Reducing false detections
- Optimizing real-time inference speed

---

## 📈 Future Improvements

- Voice emotion analysis
- Multi-person emotion tracking
- Emotion history dashboard
- Mobile app integration
- Personalized mood analytics

---

## 🎯 Applications

- Mental health monitoring
- Smart classrooms
- Human-computer interaction
- AI assistants
- Customer behavior analysis
- Security & surveillance systems

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to your branch
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Ayush Sachan**

- AI & Software Engineering Enthusiast
- Passionate about AI, Deep Learning, and Computer Vision

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!
