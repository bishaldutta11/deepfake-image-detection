import tensorflow as tf
try:
    model = tf.keras.models.load_model("best_model.h5")
    print("Inputs:", model.inputs)
    print("Outputs:", model.outputs)
except Exception as e:
    print("Error:", e)
