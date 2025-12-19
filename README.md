# Chest X-Ray Object Detection with YOLOv8

**Binary pathology detection in medical imaging using YOLOv8**

## 📋 Overview

This project implements object detection for localizing pathological findings (e.g., Atelectasis, Cardiomegaly, Effusion) in chest X-ray images using YOLOv8. The model is trained on the NIH ChexPert/ChestX-ray14 dataset with ~1,000 annotated bounding boxes.

**Task Requirements:**

- Detect and localize pathologies in X-ray images
- Handle multi-class classification (6 pathology types) or binary (pathology/normal)
- Achieve reasonable mAP on test set
- Provide inference pipeline with visualization

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create new virtual environment
python -m venv chest_env
source chest_env/bin/activate  # Linux/Mac
# or
chest_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify GPU (optional)
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Prepare Data

```bash
# Extract downloaded archives (if not done yet)
cd data/chestxray
tar -xzf images_*.tar.gz
cd ../..

# Convert CSV + lists to YOLO format
python src/data_prep.py \
    --csv-path data/BBox_List_2017.csv \
    --train-list data/train_val_list.txt \
    --test-list data/test_list.txt \
    --img-dir data/images \
    --output-dir datasets \
    --binary-mode True  # Set False for multiclass (6 classes)
```

### 3. Train Model

```bash
python src/train.py \
    --dataset datasets/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --img-size 640 \
    --device 0  # GPU device ID, or 'cpu'
```

### 4. Evaluate

```bash
python src/evaluate.py \
    --model models/best_model.pt \
    --dataset datasets/data.yaml \
    --batch-size 16 \
    --conf-threshold 0.5
```

### 5. Inference

```python
from src.utils import predict_image
from PIL import Image

# Load trained model
model_path = "models/best_model.pt"
img_path = "path/to/xray.png"

# Predict with visualization
predictions = predict_image(img_path, model_path, conf=0.5)
# Returns: bboxes, classes, confidences + saves viz
```

---

## 📊 Dataset Structure

After preparation, `datasets/` folder will have:

```
datasets/
├── images/
│   ├── train/  (~70% of annotated images)
│   ├── val/    (~15% of annotated images)
│   └── test/   (~15% from test_list.txt)
├── labels/     (YOLO txt format)
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml   (YOLOv8 config)
```

**YOLO Label Format:**

```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

---

## 🏗️ Architecture

### Data Preparation (`src/data_prep.py`)

- Parse `BBox_List_2017.csv` with bounding boxes
- Convert from `[x, y, w, h]` (pixels) → YOLO normalized format
- Split by `train_val_list.txt` / `test_list.txt`
- Option for binary (pathology/no-pathology) or multiclass (6 classes)

### Training (`src/train.py`)

- YOLOv8n (nano) or YOLOv8s (small) backbone
- SGD optimizer with cosine annealing
- Augmentation: Mosaic, Mixup, HSV, Rotation
- Validation every epoch
- Early stopping based on validation mAP

### Evaluation (`src/evaluate.py`)

- Compute mAP@0.5, mAP@0.75, mAP@0.5:0.95
- Per-class precision/recall
- Confusion matrix visualization
- Save results as CSV

### Utils (`src/utils.py`)

- Image preprocessing
- NMS post-processing
- Visualization with bboxes
- Inference batching

---

## 📈 Expected Performance

### Binary Mode (Pathology vs No-Pathology)

- mAP@0.5: ~0.75-0.85 (baseline)
- Precision: ~0.80
- Recall: ~0.75

### Multiclass Mode (6 pathology types)

- mAP@0.5: ~0.50-0.65 (if results poor, revert to binary)
- Per-class varies (Atelectasis >0.70, others ~0.40-0.60)

---

## 🔧 Configuration

Edit `config.yaml` for hyperparameters:

```yaml
model:
  backbone: yolov8s  # yolov8n, yolov8s, yolov8m
  pretrained: true

training:
  epochs: 50
  batch_size: 16
  img_size: 640
  lr0: 0.01
  lrf: 0.01  # final lr ratio
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3

augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10
  translate: 0.1
  scale: 0.5
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0

data:
  binary_mode: true  # Set to false for multiclass
  val_split: 0.15
  test_split: 0.15
```

---

## 📝 Project Structure

```
├── src/
│   ├── data_prep.py       # CSV → YOLO conversion
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation metrics
│   ├── augmentation.py    # Custom augmentations
│   ├── dataset_loader.py  # DataLoader wrapper
│   └── utils.py           # Helper functions
├── notebooks/
│   ├── 01_eda.ipynb       # Data exploration
│   └── 02_results.ipynb   # Result visualization
├── data/                  # Raw data
├── datasets/              # Processed YOLO format
├── models/                # Saved checkpoints
├── results/               # Training logs
├── README.md
├── requirements.txt
└── config.yaml
```

---

## 🎯 Development Timeline (48h)

### Day 1 (Today)

- ✅ Environment setup
- ✅ Data preparation script
- ✅ Training pipeline
- ⏳ Start training (overnight)

### Day 2

- ✅ Evaluation metrics
- ✅ Visualization & analysis
- ✅ Inference demo
- ✅ Final README & repo cleanup

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size or image size
python src/train.py --batch-size 8 --img-size 512
```

### No GPU detected

```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
# If False, install CPU-only torch
```

### Dataset mismatch

```bash
# Verify files exist
ls -la data/images/ | head
wc -l data/train_val_list.txt data/test_list.txt
```

---

## 📚 References

- **YOLOv8**: <https://github.com/ultralytics/ultralytics>
- **ChexPert Dataset**: <https://stanfordmlgroup.github.io/competitions/chexpert/>
- **Object Detection**: <https://arxiv.org/abs/1612.08242>

---

## 👤 Author

Developed as a CV/ML engineering task.

---

## 📄 License

MIT
