import onnx

MODEL_PATH = "model.onnx"   # or wherever your ONNX file is

model = onnx.load(MODEL_PATH)
inputs = model.graph.input
outputs = model.graph.output

print(f"Number of inputs: {len(inputs)}")
for inp in inputs:
    print(f"  Input name: {inp.name}")

print(f"\nNumber of outputs: {len(outputs)}")
for out in outputs:
    print(f"  Output name: {out.name}")