"""
YOLOv8 Training Script - Production Ready
Features:
- Mixed precision training (AMP)
- Learning rate scheduling (cosine annealing)
- Early stopping
- Model checkpointing (best + last)
- Metrics logging (CSV + JSON)
- Experiment tracking
- Inference on test set
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import yaml

from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# ============================================================
# SETUP LOGGING
# ============================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging to file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_dir / "training.log")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================
# TRAINING
# ============================================================

def train_yolov8(
    data_yaml: str,
    model_name: str = "yolov8n",
    epochs: int = 100,
    batch_size: int = 32,
    img_size: int = 640,
    lr: float = 0.001,
    device: str = "0",
    output_dir: str = "results",
    patience: int = 20,
    project_name: str = "chestxray-detection",
):
    """
    Train YOLOv8 model
    
    Args:
        data_yaml: Path to data.yaml
        model_name: YOLOv8 model size (nano/small/medium/large/xlarge)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        lr: Learning rate
        device: GPU device id (e.g., "0" for cuda:0, "0,1" for multi-gpu)
        output_dir: Output directory for results
        patience: Early stopping patience
        project_name: Project name for organizing results
    """
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("🚀 Starting YOLOv8 Training")
    logger.info("=" * 60)
    
    # Device setup
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    # Load model
    logger.info(f"\n📦 Loading model: yolov8{model_name}")
    model = YOLO(f"yolov8{model_name}.pt")
    
    # Training config
    train_config = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'lr0': lr,
        'device': device,
        'patience': patience,
        'save': True,
        'project': str(output_dir / project_name),
        'name': f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'exist_ok': False,
        'verbose': True,
        'plots': True,  # Save training plots
        'conf': 0.25,
        'iou': 0.45,
        'amp': True,  # Automatic Mixed Precision
        'mosaic': 1.0,  # Data augmentation
        'augment': True,
        'cache': True,
        'workers': 8,
        'seed': 42,
        'deterministic': True,
    }
    
    logger.info("\n⚙️  Training Configuration:")
    for key, value in train_config.items():
        logger.info(f"   {key}: {value}")
    
    # Save config
    config_path = output_dir / "train_config.json"
    with open(config_path, 'w') as f:
        json.dump(train_config, f, indent=2, default=str)
    logger.info(f"\n   Config saved to: {config_path}")
    
    # Train
    logger.info("\n" + "=" * 60)
    logger.info("🔥 Starting training...")
    logger.info("=" * 60)
    
    try:
        results = model.train(**train_config)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Training complete!")
        logger.info("=" * 60)
        
        # Copy best model
        best_model_src = Path(results.save_dir) / "weights" / "best.pt"
        best_model_dst = output_dir / "best_model.pt"
        
        if best_model_src.exists():
            import shutil
            shutil.copy2(best_model_src, best_model_dst)
            logger.info(f"\n📊 Best model saved to: {best_model_dst}")
        
        # Log results
        logger.info("\n📈 Training Results:")
        logger.info(f"   Results directory: {results.save_dir}")
        
        return results, best_model_dst
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        raise


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model_path: str,
    data_yaml: str,
    device: str = "0",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    """
    Evaluate model on validation set
    
    Args:
        model_path: Path to trained model
        data_yaml: Path to data.yaml
        device: GPU device
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
    """
    
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("📊 Evaluating Model")
    logger.info("=" * 60)
    
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = YOLO(model_path)
    
    # Validate
    metrics = model.val(
        data=data_yaml,
        device=device,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        batch=32,
        plots=True,
    )
    
    logger.info("\n📈 Validation Metrics:")
    logger.info(f"   mAP50: {metrics.box.map50:.4f}")
    logger.info(f"   mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"   Precision: {metrics.box.mp:.4f}")
    logger.info(f"   Recall: {metrics.box.mr:.4f}")
    
    return metrics


# ============================================================
# INFERENCE
# ============================================================

def predict_on_test_set(
    model_path: str,
    test_images_dir: str,
    output_dir: str = "results",
    conf_threshold: float = 0.25,
    device: str = "0",
):
    """
    Run inference on test set
    
    Args:
        model_path: Path to trained model
        test_images_dir: Directory with test images
        output_dir: Output directory for predictions
        conf_threshold: Confidence threshold
        device: GPU device
    """
    
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Running Inference on Test Set")
    logger.info("=" * 60)
    
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Get test images
    test_dir = Path(test_images_dir)
    image_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    
    logger.info(f"   Found {len(image_files)} test images")
    
    # Run inference
    predictions = []
    
    for img_path in tqdm(image_files, desc="Predicting"):
        results = model.predict(
            source=str(img_path),
            device=device,
            conf=conf_threshold,
            imgsz=640,
            verbose=False,
        )
        
        for result in results:
            pred_data = {
                'image': img_path.name,
                'detections': []
            }
            
            # Extract detections
            if result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    pred_data['detections'].append({
                        'bbox': box.cpu().numpy().tolist(),
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': result.names[int(cls)]
                    })
            
            predictions.append(pred_data)
    
    # Save predictions
    pred_path = output_dir / "predictions.json"
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"\n✅ Predictions saved to: {pred_path}")
    logger.info(f"   Total predictions: {len(predictions)}")
    
    return predictions


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for ChestX-ray detection')
    
    # Data
    parser.add_argument('--data-yaml', type=str, default='datasets/data.yaml',
                        help='Path to data.yaml')
    
    # Model
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Device
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device id (e.g., "0" or "0,1")')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    
    # Inference
    parser.add_argument('--predict', action='store_true',
                        help='Run inference on test set after training')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model for inference only')
    parser.add_argument('--test-dir', type=str, default='datasets/images/test',
                        help='Path to test images directory')
    
    args = parser.parse_args()
    
    # Train
    if args.model_path is None:
        results, best_model = train_yolov8(
            data_yaml=args.data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            device=args.device,
            output_dir=args.output_dir,
            patience=args.patience,
        )
        best_model = str(best_model)
    else:
        best_model = args.model_path
    
    # Evaluate
    logger = setup_logging(args.output_dir)
    metrics = evaluate_model(
        model_path=best_model,
        data_yaml=args.data_yaml,
        device=args.device,
    )
    
    # Predict
    if args.predict or args.model_path is not None:
        predictions = predict_on_test_set(
            model_path=best_model,
            test_images_dir=args.test_dir,
            output_dir=args.output_dir,
            device=args.device,
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 All done!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
