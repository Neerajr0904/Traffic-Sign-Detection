"""
preprocessing.py
----------------
Loading, resizing, normalizing, and augmenting the GTSRB (German Traffic
Sign Recognition Benchmark) dataset for CNN-based traffic sign detection.

Expected dataset layout (after downloading GTSRB):

    data/
      Train/
        0/
          00000_00000.png
          ...
        1/
        ...
        42/
      Test/
        00000.png
        ...
      Test.csv        # maps test image filenames -> ClassId

Download: https://benchmark.ini.rub.de/gtsrb_news.html
or via Kaggle: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
"""

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CLASSES = 43

# Human-readable names for each GTSRB class id (0-42)
CLASS_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing veh over 3.5 tons", 11: "Right-of-way at intersection",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Veh > 3.5 tons prohibited", 17: "No entry", 18: "General caution",
    19: "Dangerous curve left", 20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End speed + passing limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End no passing veh > 3.5 tons",
}


def load_training_data(train_dir, img_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Load images from GTSRB Train/<class_id>/*.png folder structure."""
    images, labels = [], []
    class_folders = sorted(os.listdir(train_dir), key=lambda x: int(x))

    for class_id in class_folders:
        class_path = os.path.join(train_dir, class_id)
        if not os.path.isdir(class_path):
            continue
        for file_name in os.listdir(class_path):
            if not file_name.lower().endswith((".png", ".jpg", ".ppm")):
                continue
            img_path = os.path.join(class_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(int(class_id))

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    return images, labels


def load_test_data(test_dir, test_csv_path, img_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Load GTSRB test images using the Test.csv label file."""
    df = pd.read_csv(test_csv_path)
    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(test_dir, os.path.basename(row["Path"]))
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        images.append(img)
        labels.append(int(row["ClassId"]))

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    return images, labels


def prepare_splits(images, labels, val_size=0.2, random_state=42):
    """Split into train/validation sets and one-hot encode labels."""
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=val_size, random_state=random_state, stratify=labels
    )
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    return X_train, X_val, y_train_cat, y_val_cat


def get_train_augmentor():
    """Return a Keras ImageDataGenerator configured for traffic sign augmentation."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )


def preprocess_single_image(img_path, img_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Load and preprocess a single image for inference."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    img_norm = img_resized.astype("float32") / 255.0
    return np.expand_dims(img_norm, axis=0), img
