"""
Entry Point to Call Training, Prediction and Evaluation
"""
import os
import argparse
from src.train import train_model
from src.predict import predict_image
from src.evaluate import evaluate_model

MODEL_PATH = "models/country_classifier.pth"


def main():

    parser = argparse.ArgumentParser(description="Image Classification Project")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", type=str, help="Classify an image")
    args = parser.parse_args()

    if args.train:
        train_model()  # Train and save model
    elif args.evaluate:
        if os.path.exists(MODEL_PATH):
            evaluate_model()
        else:
            print("Model not found. Train it first using --train.")
    elif args.predict:
        if os.path.exists(MODEL_PATH):
            predict_image(args.predict)
        else:
            print("Model not found. Train it first using --train.")
    else:
        print("No arguments provided. Use --train, --evaluate, or --predict <image_path>.")


if __name__ == "__main__":
    main()


