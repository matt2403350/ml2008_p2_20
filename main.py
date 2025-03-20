"""
Entry Point to Call Training, Prediction and Evaluation
"""
import os
import sys
import argparse
import subprocess
from src.predict import predict_country

MODEL_PATH = "src/models/country_classifier.pth"

PYTHON_CMD = "python3" if sys.version_info.major == 3 else "python"

def main():
    parser = argparse.ArgumentParser(description="SEA Image Classification Project:\n"
                                                    "Step 1: Extract Features\n"
                                                    "Step 2: Train & Evaluate Classical Models\n"
                                                    "Step 3: Predict Image Country\n")
    parser.add_argument("--extract_features", action="store_true", help="Extract Features from CNN Model")
    parser.add_argument("--train_classical", action="store_true", help="Train & Evaluate Classical Models")
    parser.add_argument("--predict", type=str, help="Classify an image (provide image path)")
    parser.add_argument("--model", type=str, choices=["svm", "rf", "knn"], default="rf",
                        help="Select model for prediction (default: rf)")

    args = parser.parse_args()

    if args.train_classical:
        print("Running train_classical.py...")
        subprocess.run([PYTHON_CMD, "src/train_classical.py"]) # Train and evaluate classical models
    elif args.extract_features:
        print("Running extract_features.py...")
        subprocess.run([PYTHON_CMD, "src/extract_features.py"])
    elif args.predict:
        model_type = args.model if args.model else "rf"  # Default to Random Forest
        print(f"Predicting image using {model_type.upper()} model...")
        predict_country(image_path= args.predict, model_name=model_type)
    else:
        print("Please provide an argument. Use --help for more information.")


if __name__ == "__main__":
    main()

