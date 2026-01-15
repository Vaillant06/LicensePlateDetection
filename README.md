# License Plate Detection using YOLOv8

This project implements a high-accuracy **License Plate Detection System** using the **Ultralytics YOLOv8s** model trained on a custom dataset of ~9,800 annotated license plate images (Roboflow).  
Training was done in **WSL2 Ubuntu 24.04** using an **NVIDIA RTX 4050 GPU**.

---

## 🚀 Project Overview

The goal of this project is to:

- Detect **license plates** in images, videos, and real-time webcam streams  
- Build a clean training + inference pipeline  
- Maintain a scalable dataset structure  
- Serve as Stage 1 of a full **ANPR (Automatic Number Plate Recognition)** system  

> **Stage 2 (Character Recognition)** will use a second YOLO model after cropping detected plates.

---

## 📁 Project Structure

LicensePlateDetection/
│── train/ # Training images & labels
│── valid/ # Validation images & labels
│── test/ # Test images
│── runs/ # YOLO training outputs (ignored by Git)
│── results/ # Inference outputs
│── data.yaml # Dataset configuration
│── training_license_plate.py
│── test_license_plate.py
│── .gitignore
└── README.md

## 🧠 Model Details

- **Model:** YOLOv8s  
- **Training Epochs:** 40  
- **Image Size:** 512 × 512  
- **Augmentation:** None (clean dataset)  
- **Hardware:**  
  - CPU: Intel i5 13th Gen  
  - GPU: NVIDIA RTX 4050 6GB  
  - RAM: 16GB  
  - WSL2 Ubuntu 24.04  

### 📊 Final Performance

| Metric        | Value |
|---------------|-------|
| mAP50         | **0.979** |
| mAP50-95      | **0.71** |
| Recall (R)    | ~0.95 |
| Precision (P) | ~0.94 |

Model performance is excellent for plate detection.

---

## 🛠️ Setup Instructions

### 1️⃣ Clone this repository

```bash
git clone <your-repo-url>
cd LicensePlateDetection

conda create -n ml python=3.10 -y
conda activate ml

pip install ultralytics opencv-python numpy

nvidia-smi
```

## 🏋️ Training

### The training script:

python training_license_plate.py

### Key arguments used:

epochs=50

imgsz=512

batch=8

cos_lr=True

patience=15

device=0

## 📸 Inference

Run detection on images or folders:

### Python file to  test the model
python test_license_plate.py

### Test Images Folder
test_images/

### Prediction output
results/preds/


## 🔐 .gitignore

Large files such as dataset, runs folder, result images, caches, and .env files are ignored for a clean repository.

## 📝 License

This project is for personal educational use.
Dataset used from Roboflow under their respective license.

## ⭐ Acknowledgements

Roboflow for dataset tools

Ultralytics for YOLOv8

NVIDIA for CUDA acceleration

WSL2 for Linux environment on Windows


