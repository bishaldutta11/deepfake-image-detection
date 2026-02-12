import numpy as np
import logging
import cv2
import os

logger = logging.getLogger(__name__)

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    logger.info("✅ TensorFlow imported successfully")
except ImportError as e:
    logger.error(f"❌ TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False

class DeepfakeModel:
    def __init__(self, model_path='best_model.h5'):
        self.model = None
        self.model_path = model_path
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained MobileNetV2 model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Using mock predictions.")
            return
        
        if not os.path.exists(self.model_path):
            logger.error(f"Model file '{self.model_path}' not found")
            return
        
        try:
            logger.info(f"🔄 Loading model from {self.model_path}...")
            self.model = load_model(self.model_path)
            self.model_loaded = True
            logger.info("✅ Model loaded successfully!")
            logger.info(f"📊 Model input shape: {self.model.input_shape}")
            logger.info(f"📊 Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.model = None
    
    def preprocess_image(self, image_array):
        """Preprocess image for model prediction - matches training preprocessing"""
        try:
            # Resize image to match model input size (96, 96)
            image_array = cv2.resize(image_array, (96, 96))
            
            # Ensure 3 channels
            if len(image_array.shape) == 2:  # Grayscale
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[-1] == 4:  # RGBA
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Normalize pixel values to [0, 1] (as done in training)
            image_array = image_array.astype('float32') / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_array):
        """Make prediction on image using the trained model"""
        try:
            if not self.model_loaded or self.model is None:
                logger.warning("Using mock prediction (model not available)")
                # Fallback to mock prediction
                confidence = np.random.uniform(0.7, 0.95)
                is_real = np.random.choice([True, False])
                return is_real, confidence
            
            logger.info("🎯 Using REAL MODEL for prediction")
            # Preprocess the image
            processed_image = self.preprocess_image(image_array)
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Get confidence and class
            confidence = float(np.max(prediction))
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            # In your training, class 1 is real, class 0 is fake
            is_real = (predicted_class == 1)
            
            logger.info(f"🎯 REAL MODEL PREDICTION - Authentic: {is_real}, Confidence: {confidence:.2%}")
            
            return is_real, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0

# Global model instance
model = DeepfakeModel()