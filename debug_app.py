import os
import sys

print("🔍 Debugging Deepfake App Status")
print("=" * 50)

# Check current directory and files
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.executable}")
print(f"Files in directory: {[f for f in os.listdir('.') if f.endswith(('.py', '.h5', '.js', '.html', '.css'))]}")

# Check if model file exists
model_path = 'best_model.h5'
print(f"\n📁 Model file exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    print(f"📁 Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

# Test TensorFlow import
print("\n🔄 Testing TensorFlow...")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} imported")
    
    # Test model loading directly
    from tensorflow.keras.models import load_model
    print("🔄 Testing model loading in current context...")
    try:
        test_model = load_model('best_model.h5')
        print("✅ Model loads successfully in current context!")
        print(f"📊 Input shape: {test_model.input_shape}")
        print(f"📊 Output shape: {test_model.output_shape}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        
except Exception as e:
    print(f"❌ TensorFlow import failed: {e}")

# Test our model_loader
print("\n🔄 Testing model_loader...")
try:
    from model_loader import model
    print(f"✅ model_loader imported successfully")
    print(f"📊 Model loaded: {model.model_loaded}")
    print(f"📊 Model object: {model.model}")
    
    if model.model_loaded:
        print("🎉 REAL MODEL IS LOADED AND READY!")
    else:
        print("❌ Model failed to load in model_loader")
        
except Exception as e:
    print(f"❌ model_loader import failed: {e}")

print("\n" + "=" * 50)