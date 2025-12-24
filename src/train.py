"""
YOLOv8 Training Script - Simple Edition

Run: python src/train.py
(all parameters by default, ready to run)
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO


def setup_logging(log_dir: str):
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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def train():
    """Main training function"""
    
    # ============================================================
    # PARAMETERS (CHANGE HERE IF NEEDED)
    # ============================================================
    
    data_yaml = "datasets/data.yaml"
    model_size = "n"  # n, s, m, l, x
    epochs = 40
    batch_size = 48
    img_size = 640
    learning_rate = 0.001
    device = "0"  # GPU id
    output_dir = "results"
    patience = 10  # Early stopping
    
    # ============================================================
    # INITIALIZATION
    # ============================================================
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir_path)
    
    logger.info("=" * 60)
    logger.info("Starting YOLOv8 Training")
    logger.info("=" * 60)
    
    # Check GPU
    device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_str}")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    # ============================================================
    # MODEL LOADING
    # ============================================================
    
    logger.info(f"\nLoading model: yolov8{model_size}")
    model = YOLO(f"yolov8{model_size}.pt")
    
    # ============================================================
    # TRAINING CONFIGURATION
    # ============================================================
    
    # Remove 'hyp' - not supported in new ultralytics versions
    train_config = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'lr0': learning_rate,
        'device': device,
        'patience': patience,
        'save': True,
        'project': str(output_dir_path / "detect"),
        'name': f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'exist_ok': False,
        'verbose': True,
        'plots': True,
        'amp': True,  # Automatic Mixed Precision
        'mosaic': 1.0,  # Augmentation
        'augment': True,
        'cache': False,  # No cache
        'workers': 8,
        'seed': 42,
        'deterministic': True,
        'copy_paste': 0.0,
    }
    
    logger.info("\nTraining Configuration:")
    for key, value in train_config.items():
        logger.info(f"   {key}: {value}")
    
    # Save config
    config_path = output_dir_path / "train_config.json"
    with open(config_path, 'w') as f:
        json.dump(train_config, f, indent=2, default=str)
    logger.info(f"\nConfig saved: {config_path}")
    
    # ============================================================
    # TRAINING
    # ============================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    try:
        results = model.train(**train_config)
        
        logger.info("\n" + "=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)
        
        # Copy best model
        best_model_src = Path(results.save_dir) / "weights" / "best.pt"
        best_model_dst = output_dir_path / "best_model.pt"
        
        if best_model_src.exists():
            import shutil
            shutil.copy2(best_model_src, best_model_dst)
            logger.info(f"\nBest model: {best_model_dst}")
        
        logger.info(f"Results directory: {results.save_dir}")
        
        return results, best_model_dst
        
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        raise


def evaluate(best_model_path: str):
    """Evaluate model on validation set"""
    
    output_dir_path = Path("results")
    logger = setup_logging(output_dir_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating model")
    logger.info("=" * 60)
    
    device = "0" if torch.cuda.is_available() else "cpu"
    
    model = YOLO(str(best_model_path))
    
    metrics = model.val(
        data="datasets/data.yaml",
        device=device,
        imgsz=640,
        batch=32,
        plots=True,
    )
    
    logger.info("\nValidation Metrics:")
    logger.info(f"   mAP50: {metrics.box.map50:.4f}")
    logger.info(f"   mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"   Precision: {metrics.box.mp:.4f}")
    logger.info(f"   Recall: {metrics.box.mr:.4f}")
    
    return metrics


def predict(model_path: str, test_dir: str = "datasets/images/test"):
    """Run inference on test set"""
    
    output_dir_path = Path("results")
    logger = setup_logging(output_dir_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Running inference on test set")
    logger.info("=" * 60)
    
    device = "0" if torch.cuda.is_available() else "cpu"
    predictions_dir = output_dir_path / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(str(model_path))
    
    test_path = Path(test_dir)
    image_files = sorted(test_path.glob("*.png")) + sorted(test_path.glob("*.jpg"))
    
    logger.info(f"   Found {len(image_files)} test images")
    
    # Run inference
    predictions = []
    
    for img_path in image_files:
        results = model.predict(
            source=str(img_path),
            device=device,
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
    pred_path = predictions_dir / "predictions.json"
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"\nPredictions saved: {pred_path}")
    logger.info(f"   Total predictions: {len(predictions)}")
    
    return predictions


if __name__ == "__main__":
    # Train
    results, best_model = train()
    
    # Evaluate
    metrics = evaluate(best_model)
    
    # Predict
    predictions = predict(best_model)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
