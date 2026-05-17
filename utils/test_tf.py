import numpy as np
import tensorflow as tf

model_path = "output_folder/model_dynamic_range_quant.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print("=== Inputs ===")
for i, d in enumerate(interpreter.get_input_details()):
    print(f"{i}: name={d['name']}, shape={d['shape']}, dtype={d['dtype']}")

print("\n=== Outputs ===")
for i, d in enumerate(interpreter.get_output_details()):
    print(f"{i}: name={d['name']}, shape={d['shape']}, dtype={d['dtype']}")

# Optional: test inference
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
dummy = np.random.randn(*input_details[0]['shape']).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy)
interpreter.set_tensor(input_details[1]['index'], dummy)
interpreter.invoke()
out1 = interpreter.get_tensor(output_details[0]['index'])
out2 = interpreter.get_tensor(output_details[1]['index'])
print(f"\nOutput shapes: {out1.shape}, {out2.shape}")
print(f"Sample distance: {np.linalg.norm(out1 - out2):.6f}")