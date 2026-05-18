import numpy as np
import tensorflow as tf

# Path to your quantized model
MODEL_PATH = "output_final/model_fused_sim_dynamic_range_quant.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Generate two different random images (224x224x3, normalized)
img1 = np.random.randn(1, 224, 224, 3).astype(np.float32)  # NHWC
img2 = np.random.randn(1, 224, 224, 3).astype(np.float32) * 2 + 1  # slightly different stats

# Set input tensors (the model expects two inputs: image_a, image_b)
interpreter.set_tensor(input_details[0]['index'], img1)
interpreter.set_tensor(input_details[1]['index'], img2)
interpreter.invoke()

emb1 = interpreter.get_tensor(output_details[0]['index'])
emb2 = interpreter.get_tensor(output_details[1]['index'])

dist = np.linalg.norm(emb1 - emb2)
print(f"Embedding 1 mean: {emb1.mean():.4f} std: {emb1.std():.4f}")
print(f"Embedding 2 mean: {emb2.mean():.4f} std: {emb2.std():.4f}")
print(f"Distance between different images: {dist:.4f}")

# Now test with the same image twice (should be 0 distance)
interpreter.set_tensor(input_details[0]['index'], img1)
interpreter.set_tensor(input_details[1]['index'], img1)
interpreter.invoke()
emb1_same = interpreter.get_tensor(output_details[0]['index'])
emb2_same = interpreter.get_tensor(output_details[1]['index'])
dist_same = np.linalg.norm(emb1_same - emb2_same)
print(f"Distance for identical images: {dist_same:.4f}")