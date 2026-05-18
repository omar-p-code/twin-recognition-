import onnx
from onnx import helper

model = onnx.load("model_static.onnx")
graph = model.graph

new_nodes_list = []
for node in graph.node:
    if node.op_type == "Gemm":
        A = node.input[0]
        B = node.input[1]
        C = node.input[2] if len(node.input) > 2 else None
        Y = node.output[0]

        # Create MatMul → Y_matmul
        matmul_out = Y + "_matmul"
        matmul = helper.make_node(
            "MatMul", [A, B], [matmul_out],
            name=node.name + "_matmul"
        )
        new_nodes_list.append(matmul)

        if C is not None:
            # Bias add
            add = helper.make_node(
                "Add", [matmul_out, C], [Y],
                name=node.name + "_add"
            )
            new_nodes_list.append(add)
        else:
            # No bias, rename matmul output to Y
            matmul.output[0] = Y
    else:
        new_nodes_list.append(node)

# Replace the graph's node list with the new one
graph.ClearField("node")
graph.node.extend(new_nodes_list)

onnx.save(model, "model_static_fixed.onnx")
print("✅ Gemm ops replaced (topologically safe) → model_static_fixed.onnx")