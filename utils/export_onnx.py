import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.siamese import SiameseNetwork
from config import DEVICE, IMG_SIZE

def fuse_all_linear_bn(module):
    """
    Recursively walk the model and replace every
    Sequential that contains a Linear->BatchNorm1d->ReLU pattern
    with a fused Linear->ReLU.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Sequential):
            new_layers = []
            i = 0
            layers = list(child)
            while i < len(layers):
                lin = layers[i]
                if isinstance(lin, torch.nn.Linear) and i+2 < len(layers):
                    bn = layers[i+1]
                    relu = layers[i+2]
                    if isinstance(bn, torch.nn.BatchNorm1d) and isinstance(relu, torch.nn.ReLU):
                        # Fuse
                        w = lin.weight.data
                        b = lin.bias.data if lin.bias is not None else torch.zeros(lin.out_features, device=w.device)
                        gamma = bn.weight.data
                        beta = bn.bias.data
                        mean = bn.running_mean
                        var = bn.running_var
                        eps = bn.eps

                        std = torch.sqrt(var + eps)
                        new_weight = w * (gamma / std).unsqueeze(1)
                        new_bias = (b - mean) * (gamma / std) + beta

                        fused = torch.nn.Linear(lin.in_features, lin.out_features)
                        fused.weight.data = new_weight
                        fused.bias.data = new_bias

                        new_layers.append(fused)
                        new_layers.append(relu)   # keep ReLU
                        i += 3
                        print(f"  Fused {name}[{i-3}:{i-1}]")
                        continue
                new_layers.append(layers[i])
                i += 1
            setattr(module, name, torch.nn.Sequential(*new_layers))
        else:
            # Recurse into other containers
            fuse_all_linear_bn(child)

def main():
    model = SiameseNetwork().to(DEVICE)
    ckpt = torch.load("checkpoints/checkpoint.pth", map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("🔍 Fusing all Linear+BatchNorm1d blocks...")
    fuse_all_linear_bn(model)

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    torch.onnx.export(
        model, (dummy, dummy),
        "model_fused.onnx",
        opset_version=11,
        do_constant_folding=True,
        input_names=["image_a", "image_b"],
        output_names=["embedding_a", "embedding_b"],
        dynamic_axes=None
    )
    print("✅ Exported model_fused.onnx")

if __name__ == "__main__":
    main()