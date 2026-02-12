import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

def process_uploaded_file(file_stream):
    """Process uploaded file and convert to numpy array"""
    try:
        # Read image file
        image = Image.open(file_stream)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGBA to RGB if needed
        if image_array.shape[-1] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise

def generate_indicators(is_real, confidence):
    """Generate detection indicators based on prediction"""
    if is_real:
        return [
            'Natural facial movements detected',
            'Consistent lighting throughout',
            'Normal blinking patterns',
            'Realistic skin texture',
            'Natural eye reflections',
            'Authentic facial proportions'
        ]
    else:
        indicators = [
            'Irregular facial boundaries detected',
            'Inconsistent lighting artifacts',
            'Unnatural texture patterns',
            'Temporal inconsistencies found',
            'Abnormal eye movement',
            'Audio sync discrepancies'
        ]
        
        # Adjust indicators based on confidence
        if confidence > 80:
            indicators = indicators[:4] + ['Strong evidence of manipulation'] + indicators[5:]
        elif confidence > 60:
            indicators = indicators[:3] + ['Moderate manipulation signs'] + indicators[4:]
        
        return indicators