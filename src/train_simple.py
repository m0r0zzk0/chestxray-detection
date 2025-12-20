"""
YOLOv8 Training Script - Simple Edition

Run: python src/train.py
(все параметры по дефолту, готово к запуску)
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO


def setup_logging(log_dir: str):
    """Настройка логирования"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Логирование в файл
    fh = logging.FileHandler(log_dir / "training.log")
    fh.setLevel(logging.DEBUG)
    
    # Логирование в консоль
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Формат
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def train():
    """Основная функция обучения"""
    
    # ============================================================
    # ПАРАМЕТРЫ (ИЗМЕНИ ЗДЕСЬ ЕСЛИ НУЖНО)
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
    # ИНИЦИАЛИЗАЦИЯ
    # ============================================================
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir_path)
    
    logger.info("=" * 60)
    logger.info("🚀 Начинаем обучение YOLOv8")
    logger.info("=" * 60)
    
    # Проверка GPU
    device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Устройство: {device_str}")
    logger.info(f"GPU доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    # ============================================================
    # ЗАГРУЗКА МОДЕЛИ
    # ============================================================
    
    logger.info(f"\n📦 Загружаем модель: yolov8{model_size}")
    model = YOLO(f"yolov8{model_size}.pt")
    
    # ============================================================
    # КОНФИГУРАЦИЯ ОБУЧЕНИЯ
    # ============================================================
    
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
        'cache': False,  # Без кэша!
        'workers': 8,
        'seed': 42,
        'deterministic': True,
        'copy_paste': 0.0,
    }
    
    logger.info("\n⚙️ Конфигурация обучения:")
    for key, value in train_config.items():
        logger.info(f"   {key}: {value}")
    
    # Сохраняем конфиг
    config_path = output_dir_path / "train_config.json"
    with open(config_path, 'w') as f:
        json.dump(train_config, f, indent=2, default=str)
    logger.info(f"\n💾 Конфиг сохранен: {config_path}")
    
    # ============================================================
    # ОБУЧЕНИЕ
    # ============================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("🔥 Начинаем обучение...")
    logger.info("=" * 60)
    
    try:
        results = model.train(**train_config)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Обучение завершено!")
        logger.info("=" * 60)
        
        # Копируем лучшую модель
        best_model_src = Path(results.save_dir) / "weights" / "best.pt"
        best_model_dst = output_dir_path / "best_model.pt"
        
        if best_model_src.exists():
            import shutil
            shutil.copy2(best_model_src, best_model_dst)
            logger.info(f"\n📊 Лучшая модель: {best_model_dst}")
        
        logger.info(f"📈 Директория результатов: {results.save_dir}")
        
        return results, best_model_dst
        
    except Exception as e:
        logger.error(f"\n❌ Ошибка при обучении: {e}")
        raise


def evaluate(best_model_path: str):
    """Оценка модели на валидационном наборе"""
    
    output_dir_path = Path("results")
    logger = setup_logging(output_dir_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 Оценка модели")
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
    
    logger.info("\n📈 Метрики валидации:")
    logger.info(f"   mAP50: {metrics.box.map50:.4f}")
    logger.info(f"   mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"   Precision: {metrics.box.mp:.4f}")
    logger.info(f"   Recall: {metrics.box.mr:.4f}")
    
    return metrics


def predict(model_path: str, test_dir: str = "datasets/images/test"):
    """Инференс на тестовом наборе"""
    
    output_dir_path = Path("results")
    logger = setup_logging(output_dir_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Инференс на тестовом наборе")
    logger.info("=" * 60)
    
    device = "0" if torch.cuda.is_available() else "cpu"
    predictions_dir = output_dir_path / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(str(model_path))
    
    test_path = Path(test_dir)
    image_files = sorted(test_path.glob("*.png")) + sorted(test_path.glob("*.jpg"))
    
    logger.info(f"   Найдено {len(image_files)} тестовых изображений")
    
    # Запускаем инференс
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
            
            # Извлекаем детекции
            if result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    pred_data['detections'].append({
                        'bbox': box.cpu().numpy().tolist(),
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': result.names[int(cls)]
                    })
            
            predictions.append(pred_data)
    
    # Сохраняем предсказания
    pred_path = predictions_dir / "predictions.json"
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"\n✅ Предсказания сохранены: {pred_path}")
    logger.info(f"   Всего предсказаний: {len(predictions)}")
    
    return predictions


if __name__ == "__main__":
    # Обучение
    results, best_model = train()
    
    # Оценка
    metrics = evaluate(best_model)
    
    # Инференс
    predictions = predict(best_model)
    
    print("\n" + "=" * 60)
    print("🎉 Всё готово!")
    print("=" * 60)
