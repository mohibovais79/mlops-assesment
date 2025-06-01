import numpy as np
import onnxruntime
from PIL import Image


class ImagePreprocessor:
    """
    Handles loading and preprocessing of images for the model.
    """

    def __init__(
        self,
        target_size: tuple = (224, 224),
        mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
        std: np.ndarray = np.array([0.229, 0.224, 0.225]),
    ):
        self.target_size: tuple = target_size
        self.mean: np.ndarray = mean.astype(np.float32).reshape(1, 1, 3)  # For HWC format during processing
        self.std: np.ndarray = std.astype(np.float32).reshape(1, 1, 3)

    def preprocess_pil_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Preprocesses a PIL Image object.

        """
        if pil_image.mode != "RGB":
            print(f"Image mode is '{pil_image.mode}'. Converting to 'RGB'.")
            image = pil_image.convert("RGB")
        else:
            image = pil_image

        image = image.resize(self.target_size, Image.Resampling.BILINEAR)

        img_np = np.array(image, dtype=np.float32)  # Shape: (height, width, channels)

        img_np /= 255.0

        # Normalize using mean and standard deviation values

        img_np = (img_np - self.mean) / self.std

        # Transpose from HWC to CHW (channels, height, width) as PyTorch/ONNX usually expects
        img_np = img_np.transpose((2, 0, 1))

        # Add batch dimension (1, channels, height, width)
        img_np = np.expand_dims(img_np, axis=0)

        return img_np.astype(np.float32)

    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Loads an image from a path and preprocesses it.

        """
        if not hasattr(Image.Resampling, "BILINEAR"):
            Image.Resampling.BILINEAR = Image.BILINEAR

        try:
            pil_image = Image.open(image_path)
            return self.preprocess_pil_image(pil_image)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            raise
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            raise


class ONNXModel:
    """
    Handles ONNX model loading and prediction.
    """

    def __init__(self, model_path: str):
        """
        Initializes the ONNXModel.

        """
        try:
            available_providers = onnxruntime.get_available_providers()
            chosen_provider = None

            if "CUDAExecutionProvider" in available_providers:
                print("using CUDAExecutionProvider for GPU acceleration.")
                chosen_provider = "CUDAExecutionProvider"
            elif "CPUExecutionProvider" in available_providers:
                print("using CPUExecutionProvider.")
                chosen_provider = "CPUExecutionProvider"

            self.session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
                model_path, providers=[chosen_provider]
            )
            self.input_name: str = self.session.get_inputs()[0].name
            self.output_name: str = self.session.get_outputs()[0].name
            print(f"ONNX model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading ONNX model from {model_path}: {e}")
            raise

    def predict(self, input_data_numpy: np.ndarray) -> np.ndarray:
        """
        Performs inference on the input data.
        """
        try:
            result = self.session.run([self.output_name], {self.input_name: input_data_numpy})
            return result[0]
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            raise
