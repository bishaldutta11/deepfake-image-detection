<div align="center">
  
# 🕵️‍♂️ AIDefense: Deepfake Detection Board
  
**A modern, playful, and highly interactive AI-powered deepfake detection system.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-red.svg?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Inference-orange.svg?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)

</div>

---

## 🌟 Overview

AIDefense transforms the clinical process of deepfake detection into an engaging, interactive **"Detective Evidence Board"**. Users can "pin" suspect images to a corkboard and watch as the AI forensic system analyzes the image, returning the final verdict and forensic indicators on handwritten sticky notes.

Under the hood, the system is powered by a fine-tuned **MobileNetV2** Convolutional Neural Network deployed via a lightweight Flask REST API.

---

## ✨ Key Features

* **🎨 Detective's Evidence Board UI**: A highly stylized frontend featuring vibrant 3D avatars, floating elements, and a physical "corkboard" aesthetic.
* **📸 Interactive Polaroid Uploads**: Uploaded images instantly transform into Polaroid photos "pinned" to the board.
* **📝 Dynamic Sticky Notes**: AI analysis results (confidence scores and forensic indicators) are written in handwriting and pinned directly to the board.
* **🧠 Real-Time Inference**: Connects directly to a trained TensorFlow `.h5` model to process image arrays on the fly.
* **⚡ Robust REST API**: A streamlined Flask backend that handles image normalization, prediction, and dynamic indicator generation.

---

## 🚀 Getting Started

Follow these steps to set up the Evidence Board on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
```

### 2. Set Up the Environment
It is highly recommended to use a virtual environment to manage dependencies (especially to maintain compatibility between TensorFlow and NumPy).

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Activate the environment (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
Install the required packages exactly as listed to avoid version conflicts.
```bash
pip install -r requirements.txt
```
> **Note:** The project strictly requires `numpy==1.26.4` for optimal compatibility with the loaded TensorFlow model.

### 4. Run the API Server
Start the Flask backend. The server will automatically load `best_model.h5` into memory.
```bash
python app.py
```

### 5. Start Investigating!
Open your web browser and navigate to:
**👉 `http://localhost:5000`**

---

## 🛠️ API Reference

If you wish to integrate the detection engine into other applications, you can use the built-in REST API.

### `POST /api/predict`
Upload an image for deepfake analysis.
* **Content-Type**: `multipart/form-data`
* **Body**: `file` (The image file - supports JPG, PNG, BMP, TIFF)
* **Response**:
  ```json
  {
    "prediction": "authentic",
    "confidence": 92.4,
    "indicators": [
      "Natural facial movements detected",
      "Consistent lighting throughout"
    ],
    "model_used": "live"
  }
  ```

### `GET /api/health`
Check system health and verify if the TensorFlow model is actively loaded.

### `GET /api/stats`
Retrieve mock global statistics regarding deepfake trends.

---

## 📁 Repository Structure

```text
📦 deepfake-detector
 ┣ 📂 assets/
 ┃ ┣ 📂 images/            # 3D Avatars (Hero, Detective, Data, etc.)
 ┣ 📜 app.py               # Main Flask API and TensorFlow Model Loader
 ┣ 📜 app.js               # Frontend Logic & Polaroid Animations
 ┣ 📜 index.html           # Structure of the Evidence Board UI
 ┣ 📜 main.css             # CSS Styling, Layouts, and Keyframe Animations
 ┣ 📜 requirements.txt     # Python Dependencies
 ┗ 📜 best_model.h5        # Trained MobileNetV2 Weights (ensure this is present)
```

---

## ⚙️ Automated Deployment (CI/CD)

This repository includes a GitHub Actions workflow (`.github/workflows/upload.yml`).
Whenever code is pushed to the `main` branch, the workflow will automatically run safety checks, compress the application, and upload it as a build artifact for deployment.

---
<div align="center">
  <p><i>Building trustworthy AI solutions to combat misinformation in a colorful way.</i></p>
</div>
