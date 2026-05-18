import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('../output_folder/model_fixed_float16.tflite')

# Only TFLite built‑in ops – no flex, no custom
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Dynamic range quantization (no data needed, ~4× smaller)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('../output_folder/model_builtin.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Clean TFLite model saved.")