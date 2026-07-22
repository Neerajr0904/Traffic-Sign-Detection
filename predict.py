"""
predict.py
----------
Run inference with a trained traffic sign detection model.

Usage:
    # Single image
    python predict.py --mode image --image_path path/to/sign.jpg

    # Real-time webcam detection
    python predict.py --mode webcam
"""

import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from preprocessing import CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT, preprocess_single_image

CONFIDENCE_THRESHOLD = 0.75


def parse_args():
    parser = argparse.ArgumentParser(description="Run traffic sign detection inference")
    parser.add_argument("--model_path", type=str, default="models/traffic_sign_cnn_final.keras")
    parser.add_argument("--mode", type=str, choices=["image", "webcam"], default="image")
    parser.add_argument("--image_path", type=str, help="Path to an image for single-image mode")
    parser.add_argument("--camera_index", type=int, default=0)
    return parser.parse_args()


def predict_image(model, img_path):
    input_tensor, original_img = preprocess_single_image(img_path)
    preds = model.predict(input_tensor)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    label = CLASS_NAMES.get(class_id, f"Class {class_id}")
    print(f"Prediction: {label} (class {class_id}) — confidence {confidence:.2%}")

    annotated = original_img.copy()
    cv2.putText(
        annotated, f"{label} ({confidence:.0%})", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
    cv2.imshow("Prediction", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return class_id, confidence


def run_webcam(model, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different --camera_index.")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the full frame (a real deployment would first run a
        # sign-region proposal step before classifying each crop)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img_norm = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

        preds = model.predict(img_norm, verbose=0)[0]
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])

        if confidence >= CONFIDENCE_THRESHOLD:
            label = CLASS_NAMES.get(class_id, f"Class {class_id}")
            text = f"{label} ({confidence:.0%})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Traffic Sign Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    model = load_model(args.model_path)

    if args.mode == "image":
        if not args.image_path:
            raise ValueError("--image_path is required in image mode")
        predict_image(model, args.image_path)
    else:
        run_webcam(model, args.camera_index)


if __name__ == "__main__":
    main()
