import os

import torch

from model_architecture.architecture import BasicBlock, Classifier


def convert_model_to_onnx(
    pytorch_model_instance: torch.nn.Module,
    weights_path: str,
    onnx_output_path: str,
    dummy_input_shape: tuple = (1, 3, 224, 224),
) -> None:
    """
    Converts a PyTorch model to ONNX format.

    """
    print(f"Loading PyTorch model weights from {weights_path}...")

    device = torch.device("cpu")

    try:
        pytorch_model_instance.load_state_dict(torch.load(weights_path, map_location=device))
        pytorch_model_instance.to(device)
    except Exception as e:
        print(f"Error loading weights into the model: {e}")
        return

    pytorch_model_instance.eval()

    dummy_input = torch.randn(dummy_input_shape, device=device)

    print(f"Exporting model to ONNX at {onnx_output_path}...")
    try:
        torch.onnx.export(
            pytorch_model_instance,
            dummy_input,
            onnx_output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            export_params=True,
            do_constant_folding=True,
        )
        print(f"ONNX model exported successfully to {onnx_model_output_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")


if __name__ == "__main__":
    try:
        pt_model = Classifier(block=BasicBlock, layers=[2, 2, 2, 2])
    except Exception as e:
        print(f"Could not instantiate model from pytorch_model.py: {e}")
        exit(1)

    weights_file = os.path.join("weights", "pytorch_model_weights.pth")
    if not os.path.exists(weights_file):
        print(f"Weights file not found at {weights_file}")
        exit(1)

    onnx_model_output_path = "weights/image_classifier.onnx"

    os.makedirs("weights", exist_ok=True)

    convert_model_to_onnx(pt_model, weights_file, onnx_model_output_path)
