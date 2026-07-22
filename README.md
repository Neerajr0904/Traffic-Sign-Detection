# Traffic Sign Detection using CNN 🚦

A Convolutional Neural Network (CNN) that detects and classifies German traffic signs (GTSRB dataset, 43 classes) in real time, built as part of my B.E. Computer Science & Engineering (Data Science) professional training project.

## Overview

Traffic sign recognition is a core building block of autonomous driving and driver-assistance systems. This project implements a CNN-based pipeline that:

- Preprocesses and augments traffic sign images (resizing, normalization, rotation, zoom, brightness shifts) to stay robust to real-world lighting, occlusion, and viewpoint variation
- Learns hierarchical features (edges → shapes → full sign patterns) through stacked convolution, batch-norm, and pooling blocks
- Classifies signs into 43 categories (speed limits, yield, stop, warning signs, etc.) using a softmax output layer
- Supports both single-image inference and real-time webcam detection with OpenCV

Trained and evaluated on the **German Traffic Sign Recognition Benchmark (GTSRB)**, which contains ~50,000 labeled images across 43 sign classes captured under varied lighting, angle, and occlusion conditions.

## Architecture

```
Input (32x32x3)
   │
   ▼
[Conv 32 → Conv 32 → BatchNorm → MaxPool → Dropout]   Block 1
   │
   ▼
[Conv 64 → Conv 64 → BatchNorm → MaxPool → Dropout]    Block 2
   │
   ▼
[Conv 128 → BatchNorm → MaxPool → Dropout]              Block 3
   │
   ▼
Flatten → Dense(256) → BatchNorm → Dropout → Dense(43, softmax)
```

## Project Structure

```
traffic-sign-detection-cnn/
├── data/                   # GTSRB dataset (not included — see Setup)
├── models/                 # Saved trained models
├── outputs/                # Training curves, confusion matrix
├── src/
│   ├── preprocessing.py    # Data loading, resizing, normalization, augmentation
│   ├── model.py             # CNN architecture
│   ├── train.py              # Training pipeline + evaluation
│   └── predict.py            # Single-image and real-time webcam inference
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/<your-username>/traffic-sign-detection-cnn.git
   cd traffic-sign-detection-cnn
   pip install -r requirements.txt
   ```

2. Download the GTSRB dataset (e.g. from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)) and place it under `data/` so it matches:
   ```
   data/Train/0/ ... data/Train/42/
   data/Test/
   data/Test.csv
   ```

## Usage

**Train the model:**
```bash
cd src
python train.py --train_dir ../data/Train --test_dir ../data/Test --test_csv ../data/Test.csv --epochs 30
```

**Predict on a single image:**
```bash
python predict.py --mode image --image_path ../samples/stop_sign.jpg
```

**Real-time webcam detection:**
```bash
python predict.py --mode webcam
```

## Results

After training, `outputs/` will contain:
- `training_curves.png` — accuracy/loss over epochs
- `confusion_matrix.png` — per-class classification performance

*(Fill in your own measured test accuracy here once training completes.)*

## Tech Stack

Python · TensorFlow/Keras · OpenCV · NumPy · Pandas · scikit-learn · Matplotlib/Seaborn

## Future Work

- Explore lightweight architectures (MobileNet) for embedded/edge deployment
- Add a proper sign-region proposal step before classification for cluttered scenes
- Transfer learning from ImageNet-pretrained backbones
- Ensemble/ensemble-distillation for improved generalization

## License

MIT
