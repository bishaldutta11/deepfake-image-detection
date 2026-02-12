import os
print("🔍 Testing model loading...")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

if 'best_model.h5' in os.listdir('.'):
    print("✅ best_model.h5 found!")
    file_size = os.path.getsize('best_model.h5')
    print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
else:
    print("❌ best_model.h5 not found in current directory!")

print("\n🔄 Testing TensorFlow import...")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} imported successfully")
    
    print("\n🔄 Testing model loading...")
    from tensorflow.keras.models import load_model
    
    try:
        model = load_model('best_model.h5')
        print("🎉 Model loaded successfully!")
        print(f"📊 Input shape: {model.input_shape}")
        print(f"📊 Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        
        print("\n🔄 Trying with compile=False...")
        try:
            model = load_model('best_model.h5', compile=False)
            print("✅ Model loaded with compile=False!")
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            
except Exception as e:
    print(f"❌ TensorFlow import failed: {e}")