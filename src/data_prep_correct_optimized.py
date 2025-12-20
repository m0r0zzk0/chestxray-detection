"""
Data preparation: CORRECT approach - OPTIMIZED
Strategy:
- Train: Normal images (no boxes) - model learns background
- Val: MIX of pathology (90) + normal (300) - can see actual objects
- Test: All pathology (880) - final evaluation
This ensures model learns AND validation metrics work!
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse
import shutil
import yaml


def normalize_bbox(x, y, w, h, img_width, img_height):
    """Convert bbox from [x,y,w,h] pixels to YOLO normalized [x_c,y_c,w_norm,h_norm]"""
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    w_norm = np.clip(w_norm, 0, 1)
    h_norm = np.clip(h_norm, 0, 1)
    
    return x_center, y_center, w_norm, h_norm


def prepare_dataset(
    csv_path,
    train_list_path,
    test_list_path,
    img_dir,
    output_dir,
    binary_mode=True,
    max_train_normal=None,  # Limit normal images for faster training
):
    """
    Prepare YOLO dataset CORRECTLY.
    
    Strategy:
    - Train: ~5k normal images + 700 pathology (optimized for speed)
    - Val: 90 pathology + 300 normal (validation WITH boxes)
    - Test: 790 pathology (final evaluation)
    
    Set max_train_normal=None to use ALL 86k images for production
    """
    
    print("=" * 60)
    print("🔄 Preparing CORRECT dataset for YOLOv8...")
    if max_train_normal:
        print(f"⚡ Optimized: max {max_train_normal} normal images for speed")
    else:
        print("📊 Production: using ALL available normal images")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    print("\n📖 Loading data...")
    
    df = pd.read_csv(csv_path)
    annotated_images = list(df['Image Index'].unique())
    print(f"   Annotated images (CSV): {len(annotated_images)}")
    
    with open(train_list_path, 'r') as f:
        train_val_images = set(line.strip() for line in f if line.strip())
    print(f"   Train/Val list total: {len(train_val_images)}")
    
    normal_images = list(train_val_images - set(annotated_images))
    print(f"   Normal images available: {len(normal_images)}")
    
    # BUILD CLASS MAPPING
    if binary_mode:
        class_names = ["pathology"]
        class_map = {label: 0 for label in df['Finding Label'].unique()}
        print(f"\n📌 Binary mode: pathology class")
    else:
        class_names = sorted(df['Finding Label'].unique().tolist())
        class_map = {label: idx for idx, label in enumerate(class_names)}
        print(f"📌 Multiclass mode: {len(class_names)} classes")
    
    print(f"   Classes: {class_names}")
    
    # GROUP ANNOTATIONS
    print("\n🗂️  Grouping annotations by image...")
    annotations = defaultdict(list)
    
    for idx, row in df.iterrows():
        img_name = row['Image Index']
        label = row['Finding Label']
        x = float(row.iloc[2])
        y = float(row.iloc[3])
        w = float(row.iloc[4])
        h = float(row.iloc[5])
        
        annotations[img_name].append({
            'class': class_map[label],
            'bbox': [x, y, w, h]
        })
    
    # SPLIT STRATEGY
    print("\n📊 Splitting data...")
    np.random.shuffle(annotated_images)
    np.random.shuffle(normal_images)
    
    # Val/Test: 90/790 pathology
    pathology_val = annotated_images[:90]
    pathology_test = annotated_images[90:180]
    pathology_train = annotated_images[180:]  # Rest to train
    
    # Normal split
    normal_val = normal_images[:300]
    if max_train_normal:
        normal_train = normal_images[300:300+max_train_normal]
    else:
        normal_train = normal_images[300:]  # All remaining for production
    
    print(f"   Train: {len(pathology_train)} pathology + {len(normal_train)} normal = {len(pathology_train) + len(normal_train)} total")
    print(f"   Val: {len(pathology_val)} pathology + {len(normal_val)} normal = {len(pathology_val) + len(normal_val)} total")
    print(f"   Test: {len(pathology_test)} pathology")
    
    # PROCESS IMAGES
    print("\n⚙️  Processing and copying images...")
    print("   (This may take 15-60 minutes depending on dataset size)")
    
    img_dir = Path(img_dir)
    split_stats = {"train": 0, "val": 0, "test": 0}
    missing = []
    
    def process_image_list(img_names, split, has_annotations):
        """Helper to process list of images"""
        for img_name in tqdm(img_names, desc=f"{split.upper()} images", leave=False):
            img_path = img_dir / img_name
            
            if not img_path.exists():
                missing.append(img_name)
                continue
            
            # Read and convert grayscale to RGB
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                missing.append(img_name)
                continue
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w = img_rgb.shape[:2]
            
            # Create label file
            label_filename = img_name.replace('.png', '.txt')
            label_path = output_path / "labels" / split / label_filename
            
            if has_annotations:
                with open(label_path, 'w') as f:
                    for ann in annotations[img_name]:
                        class_id = ann['class']
                        x, y, bbox_w, bbox_h = ann['bbox']
                        x_norm, y_norm, w_norm, h_norm = normalize_bbox(x, y, bbox_w, bbox_h, w, h)
                        f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            else:
                label_path.touch()
            
            # Copy image (converted to RGB)
            dest_img = output_path / "images" / split / img_name
            try:
                cv2.imwrite(str(dest_img), img_rgb)
                split_stats[split] += 1
            except Exception as e:
                print(f"⚠️  Failed to copy {img_name}: {e}")
    
    # Process each split
    process_image_list(normal_train, "train", has_annotations=False)
    process_image_list(pathology_train, "train", has_annotations=True)
    
    process_image_list(normal_val, "val", has_annotations=False)
    process_image_list(pathology_val, "val", has_annotations=True)
    
    process_image_list(pathology_test, "test", has_annotations=True)
    
    # STATISTICS
    print("\n" + "=" * 60)
    print("📊 Dataset Statistics")
    print("=" * 60)
    print(f"Train images: {split_stats['train']}")
    print(f"Val images:   {split_stats['val']}")
    print(f"Test images:  {split_stats['test']}")
    print(f"Total: {sum(split_stats.values())}")
    
    if missing:
        print(f"\n⚠️  Missing images: {len(missing)}")
    
    # VERIFY
    print("\n✔️  Verifying dataset structure...")
    for split in ["train", "val", "test"]:
        imgs = list((output_path / "images" / split).glob("*.png"))
        lbls = list((output_path / "labels" / split).glob("*.txt"))
        non_empty = sum(1 for lbl in lbls if lbl.stat().st_size > 0)
        print(f"   {split.upper()}: {len(imgs)} images, {len(lbls)} labels ({non_empty} with boxes)")
    
    # CREATE DATA.YAML
    print("\n📝 Creating data.yaml...")
    
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': str((output_path / 'images' / 'train').absolute()),
        'val': str((output_path / 'images' / 'val').absolute()),
        'test': str((output_path / 'images' / 'test').absolute()),
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"   Saved to: {yaml_path}")
    
    print("\n" + "=" * 60)
    print("✅ CORRECT dataset preparation complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Prepare CORRECT YOLO dataset')
    parser.add_argument('--csv-path', type=str, default='data/annotations.csv')
    parser.add_argument('--train-list', type=str, default='data/train_val_list.txt')
    parser.add_argument('--test-list', type=str, default='data/test_list.txt')
    parser.add_argument('--img-dir', type=str, default='data/images')
    parser.add_argument('--output-dir', type=str, default='datasets')
    parser.add_argument('--max-train-normal', type=int, default=5000,
                        help='Max normal images for training (set to 0 for ALL)')
    
    args = parser.parse_args()
    
    prepare_dataset(
        csv_path=args.csv_path,
        train_list_path=args.train_list,
        test_list_path=args.test_list,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        max_train_normal=args.max_train_normal if args.max_train_normal > 0 else None,
    )


if __name__ == '__main__':
    main()
