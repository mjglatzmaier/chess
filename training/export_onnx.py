"""
Export a trained chess transformer to ONNX format.

The exported model can be loaded by the C++ engine via ONNX Runtime
or used as a reference for custom C++ inference implementation.

Usage:
    python export_onnx.py models/round_0.pt --output models/round_0.onnx
"""

import argparse

import torch
import numpy as np

from model import ChessTransformer
from config import ModelConfig


def export_to_onnx(model_path: str, output_path: str) -> None:
    """Export a trained model to ONNX format."""

    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu")
    config = ModelConfig(**checkpoint["config"])
    model = ChessTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model: {model.count_parameters():,} parameters")

    # Create dummy input
    dummy_input = torch.randn(1, 64, 25)

    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["board"],
        output_names=["value", "policy"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "value": {0: "batch_size"},
            "policy": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Validate
    import onnxruntime as ort

    session = ort.InferenceSession(output_path)
    test_input = np.random.randn(4, 64, 25).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_value, pt_policy = model(torch.from_numpy(test_input))

    # ONNX inference
    onnx_outputs = session.run(None, {"board": test_input})
    onnx_value = onnx_outputs[0]
    onnx_policy = onnx_outputs[1]

    # Compare
    value_diff = np.abs(pt_value.numpy() - onnx_value).max()
    policy_diff = np.abs(pt_policy.numpy() - onnx_policy).max()

    print(f"Validation:")
    print(f"  Value  max diff: {value_diff:.8f}")
    print(f"  Policy max diff: {policy_diff:.8f}")

    if value_diff < 1e-5 and policy_diff < 1e-5:
        print("  ✓ ONNX output matches PyTorch")
    else:
        print("  ⚠ Outputs differ — check quantization settings")

    print(f"\nExported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("model", help="Path to .pt checkpoint")
    parser.add_argument("-o", "--output", default=None, help="Output .onnx path")
    args = parser.parse_args()

    output = args.output or args.model.replace(".pt", ".onnx")
    export_to_onnx(args.model, output)


if __name__ == "__main__":
    main()
