import argparse
import json
import os
import time

import requests


def make_prediction_request(image_path: str, api_url: str, api_key: str) -> tuple:
    """
    Sends an image to the deployed API endpoint and returns the prediction.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}

            start_time = time.time()
            response = requests.post(api_url, files=files, headers=headers, timeout=30)
            end_time = time.time()

            latency = end_time - start_time
            print(f"Request to {api_url} took {latency:.3f} seconds.")

            response.raise_for_status()
            return response.json(), latency

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request: {req_err}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response: {response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, None


def run_server_tests(api_url: str, api_key: str, run_preset_flag: bool, image_to_test: str = None):
    """
    Main function to run tests against the deployed server.
    """
    print(f"--- Testing Deployed Server at: {api_url} ---")

    if not api_url:
        print("API URL is required. Please provide --api_url.")
        return

    if image_to_test:
        print(f"\nTesting with specific image: {image_to_test}")
        result, latency = make_prediction_request(image_to_test, api_url, api_key)
        if result:
            print(f"Prediction for {os.path.basename(image_to_test)}:")
            print(f"  Predicted Class ID: {result.get('predicted_class_id')}")
            print(f"  Probabilities (first 5): {result.get('probabilities', [])[:5]}")
            print(f"  Latency: {latency:.3f}s")
            if latency and latency > 3.0:
                print(f"  WARNING: Latency ({latency:.3f}s) exceeds target of 2-3 seconds.")
        else:
            print(f"Failed to get prediction for {image_to_test}")

    if run_preset_flag:
        print("\n--- Running Preset Custom Tests ---")
        test_images_info = {"images/n01440764_tench.jpeg": 0, "images/n01667114_mud_turtle.jpeg": 35}
        preset_tests_passed = 0
        total_preset_tests = len(test_images_info)

        for image_path, expected_id in test_images_info.items():
            if not os.path.exists(image_path):
                print(f"Preset test image not found: {image_path}. Skipping.")
                total_preset_tests -= 1
                continue

            print(f"Testing preset image: {os.path.basename(image_path)} (Expected ID: {expected_id})")
            result, latency = make_prediction_request(image_path, api_url, api_key)
            if result and "predicted_class_id" in result:
                predicted_id = result["predicted_class_id"]
                print(f"  Got Predicted Class ID: {predicted_id}, Latency: {latency:.3f}s")
                if predicted_id == expected_id:
                    print(f"  Test PASSED for {os.path.basename(image_path)}")
                    preset_tests_passed += 1
                else:
                    print(f"  Test FAILED: Expected {expected_id}, Got {predicted_id}")
                if latency and latency > 3.0:
                    print(
                        f"  WARNING: Latency ({latency:.3f}s) for {os.path.basename(image_path)} exceeds target of 2-3 seconds."
                    )
            else:
                print(f"  Test FAILED for {os.path.basename(image_path)}: No valid result from server.")

        print(f"\nPreset Test Summary: {preset_tests_passed}/{total_preset_tests} passed.")

    print("\n--- Additional Platform Tests ---")
    health_url = api_url.replace("/predict", "/health")
    if "/predict" not in api_url:
        base_url = api_url.rstrip("/")
        health_url = f"{base_url}/health"

    print(f"Performing health check on: {health_url}")
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.get(health_url, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"Health check PASSED. Status: {response.status_code}, Response: {response.json()}")
    except Exception as e:
        print(f"Health check FAILED: {e}")

    print("\nTesting with a non-image file (expecting client or server error)...")
    non_image_file = "requirements.txt"
    if os.path.exists(non_image_file):
        result, _ = make_prediction_request(non_image_file, api_url, api_key)
        if result is None:
            print("Server correctly handled non-image file (request failed as expected or server returned error).")

    else:
        print(f"Skipping non-image file test: {non_image_file} not found.")

    print("\n--- Server Test Run Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the deployed image classification API on Cerebrium.")
    parser.add_argument(
        "--api_url",
        type=str,
        required=True,
        help="API URL of the deployed model (e.g., https://run.cerebrium.ai/your-model-version/predict).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="API Key for authorization",
    )
    parser.add_argument("--image_path", type=str, help="Path to a single image for testing.")
    parser.add_argument(
        "--run_preset_tests", action="store_true", help="Run tests with preset images (tench, mud_turtle)."
    )

    args = parser.parse_args()

    api_key_to_use = args.api_key or os.getenv("CEREBRIUM_API_KEY")
    if not api_key_to_use:
        print("Warning: API Key not provided.")

    if not args.image_path and not args.run_preset_tests:
        print("Please specify either an --image_path or --run_preset_tests (or both).")
        parser.print_help()
    else:
        run_server_tests(args.api_url, api_key_to_use, args.run_preset_tests, args.image_path)
