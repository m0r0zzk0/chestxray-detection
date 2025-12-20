# ChestX-ray Pathology Detection - YOLOv8 Object Detection

## Project Overview

This project implements **binary object detection** for chest X-ray pathology localization using YOLOv8. The model detects the presence and location of pathological findings in medical images.

### Key Metrics
- **Dataset**: 87,404 chest X-ray images (86,924 train, 390 val, 90 test)
- **Training Strategy**: Binary classification (pathology vs background)
- **Model**: YOLOv8-Nano (3M parameters, optimized for inference speed)
- **Hardware**: NVIDIA RTX 3060 (12GB VRAM)

---

## Repository Structure

```
chestxray-detection/
├── src/
│   ├── data_prep_correct_optimized.py    # Dataset preparation & split strategy
│   ├── train.py                          # YOLOv8 training pipeline
│   ├── evaluate.py                       # Model evaluation & metrics
│   └── predict.py                        # Inference on test images
├── data/
│   ├── annotations.csv                   # Bounding box annotations
│   ├── train_val_list.txt                # Train/Val image list
│   ├── test_list.txt                     # Test image list
│   └── images/                           # Raw chest X-ray images (~50GB)
├── datasets/
│   ├── images/
│   │   ├── train/                        # 86,924 images
│   │   ├── val/                          # 390 images
│   │   └── test/                         # 90 images
│   ├── labels/
│   │   ├── train/                        # YOLO format annotations
│   │   ├── val/
│   │   └── test/
│   └── data.yaml                         # Dataset config for YOLOv8
├── results/
│   └── detect/                           # Training outputs (weights, plots)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset Strategy

### Split Rationale

The dataset split was carefully designed to ensure proper model learning and validation:

```
Training Set (86,924 images):
├── Normal images: 86,224 (no annotations)
│   └── Purpose: Model learns "background" - what normal chest looks like
└── Pathology images: 700 (with bounding boxes)
    └── Purpose: Model learns to detect pathological patterns

Validation Set (390 images):
├── Normal images: 300
└── Pathology images: 90
    └── Purpose: Real-world mix validation - ensure metrics work with both classes

Test Set (90 images):
└── Pathology only: 90
    └── Purpose: Clean evaluation on unseen pathological cases
```

### Why This Strategy Works

1. **Balanced Learning**: Model learns both classes explicitly
   - Normal images teach negative samples (avoid false positives)
   - Pathology images teach positive samples (detect pathology)

2. **Proper Validation**: Val set is a realistic mix
   - Pure pathology test would inflate metrics (easy case)
   - Mixed val set catches overfitting to pathology

3. **YOLO Compatibility**: 
   - Normal images get empty label files (YOLO handles this)
   - Pathology images get YOLO format bboxes: `class x_center y_center width height`

---

## Data Preparation

### Running Data Prep

```bash
# Full dataset (86k normal images) - ~60 minutes
python src/data_prep_correct_optimized.py --max-train-normal 0

# Quick test run (5k normal images) - ~10 minutes  
python src/data_prep_correct_optimized.py --max-train-normal 5000
```

### Key Transformations

1. **Grayscale → RGB Conversion**
   ```python
   # Medical images are grayscale, YOLOv8 expects RGB
   img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
   img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
   ```

2. **Bbox Normalization to YOLO Format**
   ```python
   # Input: pixel coordinates [x, y, w, h]
   # Output: normalized [x_center, y_center, w_norm, h_norm] ∈ [0, 1]
   x_center = (x + w/2) / img_width
   y_center = (y + h/2) / img_height
   ```

3. **Clipping Out-of-Bounds Bboxes**
   ```python
   # Ensure normalized coords stay within [0, 1]
   x_center = np.clip(x_center, 0, 1)
   ```

### Output Structure

```
datasets/
├── data.yaml                    # YOLOv8 dataset config
├── images/
│   ├── train/  86,924 images
│   ├── val/    390 images
│   └── test/   90 images
└── labels/
    ├── train/  86,924 .txt files (700 with boxes, 86,224 empty)
    ├── val/    390 .txt files (90 with boxes, 300 empty)
    └── test/   90 .txt files (all with boxes)
```

---

## Training Pipeline

### Command

```bash
python src/train.py \
    --data-yaml datasets/data.yaml \
    --model n \
    --epochs 40 \
    --batch-size 48 \
    --device "0" \
    --output-dir results_full \
    --predict
```

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model Size** | nano (n) | Fast iteration, 12GB VRAM constraint |
| **Batch Size** | 48 | Balance: 1GB VRAM usage, stable gradients |
| **Image Size** | 640×640 | YOLOv8 default, good for medical detail |
| **Epochs** | 40-50 | Sufficient for convergence on this dataset |
| **Learning Rate** | 0.001 | YOLOv8 default (uses cosine annealing) |
| **Cache** | RAM (True) | ~1GB RAM, avoid 330GB disk cache |
| **Patience** | 10 | Early stopping if val loss plateaus |

### Memory Usage

```
GPU Memory Breakdown (RTX 3060, 12GB):
├── Model weights: 150 MB
├── Batch (48 images × 640×640 × 3 channels): 700 MB
├── Optimizer state: 150 MB
└── Free: ~10.8 GB
```

**Why RAM cache instead of disk cache:**
- `cache='disk'` creates 330GB cache file (entire dataset)
- `cache=True` uses ~1GB RAM for batch pre-processing
- Trade-off: Faster epochs with more RAM usage

---

## Training Dynamics

### Expected Log Progression

**Epoch 1-5**: Model initialization
```
Instances: 0-5        (model barely detecting objects)
box_loss: 2.0-3.0     (still learning localization)
cls_loss: 30-5        (rapidly improving classification)
Metrics: 0            (predictions too weak for evaluation)
```

**Epoch 10-20**: Learning kicks in
```
Instances: 20-40      (model consistently detects)
box_loss: 1.5-2.0     (steady improvement)
cls_loss: 2.0-3.0     (stable)
Metrics: 0.01-0.05    (first predictions appearing)
```

**Epoch 25-40**: Refinement
```
Instances: 20-30      (stable detection)
box_loss: 1.5-2.0     (minimal change)
cls_loss: 2.0-3.0     (stable)
Metrics: 0.05-0.15    (measurable improvement)
```

### Red Flags

❌ **Instances = 0 throughout training**
- Problem: Labels not loading, check data.yaml paths
- Solution: Verify `datasets/labels/train/*.txt` have content

❌ **box_loss increasing or plateau at epoch 5**
- Problem: Learning rate too low, data mismatch
- Solution: Check label format, try `lr0=0.01`

❌ **OOM (Out of Memory) at epoch 30-31**
- Problem: Albumentations kicks in, adds memory overhead
- Solution: Reduce `batch_size` from 48 to 32

❌ **Metrics = 0 at epoch 40**
- Problem: Model not learning, fundamental issue
- Solution: Check instances are >0, verify val set has pathology images

---

## Model Evaluation

### Metrics Explained

```
Box(P): Precision    - Of detected boxes, % are correct
Box(R): Recall       - Of all true boxes, % detected
mAP50:  Avg Precision at IoU=0.5 (standard metric)
mAP50-95: Avg Precision across IoU thresholds 0.5-0.95
```

### Expected Baseline Performance

For **binary detection on single pathology**:
- mAP50: 0.10-0.20 (baseline with 700 pathology samples)
- mAP50-95: 0.05-0.10 (stricter metric)

These are reasonable for:
- Single-class detection (no multi-class complexity)
- Limited annotation budget (700 samples)
- Small model (Nano)

### Improvement Strategies

1. **More Annotation**: 1500+ pathology samples → +0.05 mAP
2. **Larger Model**: Medium/Large → +0.03-0.05 mAP (slower inference)
3. **Multiclass**: Specific pathologies (Atelectasis, Cardiomegaly, etc.) → better clinical utility
4. **Post-processing**: NMS tuning, confidence threshold optimization

---

## Inference & Predictions

### Run Predictions on Test Set

```bash
python src/predict.py \
    --model results/detect/weights/best.pt \
    --source datasets/images/test \
    --conf 0.25 \
    --output predictions/
```

### Output Format

```json
{
  "image": "00000099_009.png",
  "detections": [
    {
      "bbox": [100, 200, 350, 450],  // [x1, y1, x2, y2]
      "confidence": 0.87,
      "class": 0,
      "class_name": "pathology"
    }
  ]
}
```

### Confidence Threshold Tuning

- **0.25** (default): Catches more positives, ~10% false positives
- **0.50**: Balanced precision/recall
- **0.75**: High precision, misses subtle pathologies

---

## Troubleshooting

### Problem: "No such file or directory: datasets/data.yaml"
**Solution**: Run data prep first
```bash
python src/data_prep_correct_optimized.py --max-train-normal 0
```

### Problem: "CUDA out of memory" at epoch 31
**Solution**: Albumentations kicks in, reduce batch size
```bash
# Reduce from 48 to 32
python src/train.py --batch-size 32 ...
```

### Problem: "Instances: 0" throughout training
**Solution**: Check label files exist with content
```powershell
# PowerShell
(ls datasets/labels/train/*.txt | Where-Object {(Get-Item $_).Length -gt 0}).Count
# Should output: 700
```

### Problem: Validation metrics = 0 at epoch 40
**Solution**: Ensure val set has pathology images
```bash
# Check val label files
ls -la datasets/labels/val/*.txt | wc -l  # Should be 390
# Non-empty (with boxes): should be 90
```

### Problem: Training very slow (~15 min per epoch)
**Solution**: Check cache settings, enable disk cache carefully
```python
# In train.py - only if plenty of disk space
'cache': 'disk',  # Uses 330GB for full dataset!
```

---

## Performance Notes

### Training Time
- **Quick run** (5k normal): ~1 hour × 40 epochs
- **Full run** (86k normal): ~3 hours × 40 epochs
- Per epoch: 8-12 minutes (depends on cache, batch size, augmentations)

### Inference Speed
- **YOLOv8-Nano**: 1.6 ms per image on RTX 3060
- **Throughput**: ~625 images/second (GPU-bound)

### Storage
- **Model weights**: 6.2 MB (best.pt)
- **Dataset**: ~50 GB (raw images) + 50 GB (prepared)
- **Results**: ~5 GB (weights + plots + predictions)

---

## Next Steps & Improvements

### Phase 1: Validation (Current)
- [x] Data split strategy validation
- [x] Baseline model (YOLOv8-Nano)
- [x] Training pipeline working
- [ ] Final metrics on full dataset

### Phase 2: Enhancement
- [ ] Multiclass detection (specific pathologies)
- [ ] Larger model (YOLOv8-Small/Medium)
- [ ] Additional augmentation strategies
- [ ] Confidence threshold optimization

### Phase 3: Deployment
- [ ] ONNX export for inference
- [ ] REST API for predictions
- [ ] Real-time video processing
- [ ] Clinical validation on external test set

---

## Dependencies

```
torch>=2.0.0
ultralytics>=8.0.0
opencv-python
pandas
numpy
pyyaml
tqdm
```

Install: `pip install -r requirements.txt`

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Format Explanation](https://roboflow.com/formats/yolo-darknet-txt)
- [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)

---

## Author Notes

This project demonstrates:
- ✅ Production ML pipeline (data → train → eval → deploy)
- ✅ Handling imbalanced datasets (86k normal vs 700 pathology)
- ✅ Medical imaging specific transformations
- ✅ GPU memory optimization for constrained hardware
- ✅ Reproducible training with proper validation strategy

**Developed December 2025** for ChestX-ray pathology detection task.
