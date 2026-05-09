# Deepfake Detection System

A web-based application to detect deepfakes and manipulated images using a trained Deep Learning model (MobileNetV2).

## Features
- **Frontend Interface:** Clean, responsive UI for uploading images and viewing analysis.
- **Backend API:** Built with Flask to handle image processing and model inference.
- **Real-Time Analysis:** Uses a trained TensorFlow model (`best_model.h5`) to classify images as "authentic" or "deepfake", returning a confidence score.
- **Automated Insights:** Generates dynamic indicators based on the model's confidence and prediction.

## Requirements
- Python 3.8+
- Requirements listed in `requirements.txt`

## Local Setup

1. **Clone or Download** this repository.
2. **Navigate** to the project directory:
   ```bash
   cd path/to/deepfake
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Server**:
   ```bash
   python app.py
   ```
5. **Access the Application**: Open your browser and navigate to `http://localhost:5000`.

## API Endpoints

### 1. `GET /api/health`
Check if the API and model are operational.
- **Response**: JSON object with system status.

### 2. `POST /api/predict`
Upload an image to get a deepfake analysis.
- **Payload**: `multipart/form-data` containing an image `file`.
- **Response**: JSON containing the prediction, confidence score, and specific visual indicators.

### 3. `GET /api/stats`
Returns system usage statistics.

## Deployment / CI/CD
This project includes a GitHub Actions workflow (`.github/workflows/upload.yml`) that automatically tests the code, packages the application, and uploads it as a build artifact whenever you push to the `main` branch.
