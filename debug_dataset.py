"""
Dataset Debug Script for YOLOv8 Training

Checks if dataset, labels, and data.yaml are correctly configured.
Run this BEFORE training to catch issues early.
"""

import os
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def check_data_yaml(yaml_path):
    """Verify data.yaml has correct paths and structure."""
    print("\n" + "="*60)
    print("📋 Checking data.yaml...")
    print("="*60)
    
    if not os.path.exists(yaml_path):
        print(f"❌ data.yaml not found at {yaml_path}")
        return False
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"✅ data.yaml loaded successfully")
        print(f"   Names: {data.get('names')}")
        print(f"   Classes: {data.get('nc')}")
        print(f"   Path: {data.get('path')}")
        
        # Check paths exist
        for split in ['train', 'val', 'test']:
            path = data.get(split)
            if path:
                exists = os.path.exists(path)
                status = "✅" if exists else "❌"
                print(f"   {status} {split}: {path}")
        
        return True
    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")
        return False


def check_images_and_labels(images_dir, labels_dir, split='train'):
    """Verify images and labels match and are correctly formatted."""
    print("\n" + "="*60)
    print(f"🖼️  Checking {split} images and labels...")
    print("="*60)
    
    images_path = Path(images_dir) / split
    labels_path = Path(labels_dir) / split
    
    if not images_path.exists():
        print(f"❌ Images directory not found: {images_path}")
        return False
    
    if not labels_path.exists():
        print(f"❌ Labels directory not found: {labels_path}")
        return False
    
    # Count files
    image_files = sorted([f.name for f in images_path.glob('*.png')] + 
                        [f.name for f in images_path.glob('*.jpg')])
    label_files = sorted([f.name for f in labels_path.glob('*.txt')])
    
    print(f"   Images: {len(image_files)}")
    print(f"   Labels: {len(label_files)}")
    
    # Check matching
    image_stems = set(f.rsplit('.', 1)[0] for f in image_files)
    label_stems = set(f.rsplit('.', 1)[0] for f in label_files)
    
    matches = image_stems & label_stems
    only_images = image_stems - label_stems
    only_labels = label_stems - image_stems
    
    print(f"   ✅ Matching pairs: {len(matches)}")
    
    if only_images:
        print(f"   ⚠️  Images without labels: {len(only_images)} (expected for normal images)")
    if only_labels:
        print(f"   ❌ Labels without images: {len(only_labels)}")
        for label in list(only_labels)[:5]:
            print(f"      - {label}.txt")
    
    # Check label file sizes
    empty_labels = 0
    nonempty_labels = 0
    
    for label_file in labels_path.glob('*.txt'):
        if label_file.stat().st_size == 0:
            empty_labels += 1
        else:
            nonempty_labels += 1
    
    print(f"   Empty labels (no objects): {empty_labels}")
    print(f"   Non-empty labels (with objects): {nonempty_labels}")
    
    if nonempty_labels == 0:
        print(f"   ❌ WARNING: No objects found in labels!")
        return False
    
    return True


def check_label_format(labels_dir, split='train'):
    """Verify label format is correct YOLO format."""
    print("\n" + "="*60)
    print(f"📝 Checking label format ({split})...")
    print("="*60)
    
    labels_path = Path(labels_dir) / split
    
    # Find non-empty labels
    nonempty_labels = [f for f in labels_path.glob('*.txt') if f.stat().st_size > 0]
    
    if not nonempty_labels:
        print("❌ No non-empty label files found")
        return False
    
    print(f"   Checking {min(5, len(nonempty_labels))} sample labels...")
    
    errors = []
    for label_file in nonempty_labels[:5]:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines):
                parts = line.strip().split()
                
                if len(parts) != 5:
                    errors.append(f"{label_file.name}:{line_num+1} - Expected 5 values, got {len(parts)}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    
                    # Check bounds
                    for i, coord in enumerate(coords):
                        if coord < 0 or coord > 1:
                            coord_name = ['x_center', 'y_center', 'width', 'height'][i]
                            errors.append(f"{label_file.name}:{line_num+1} - {coord_name} out of bounds: {coord}")
                except ValueError as e:
                    errors.append(f"{label_file.name}:{line_num+1} - Parse error: {e}")
            
            print(f"   ✅ {label_file.name}: {len(lines)} objects, format OK")
        
        except Exception as e:
            errors.append(f"{label_file.name} - Read error: {e}")
    
    if errors:
        print(f"\n   ❌ Found {len(errors)} format errors:")
        for error in errors[:10]:
            print(f"      - {error}")
        return False
    else:
        print(f"   ✅ Label format is correct")
        return True


def check_image_quality(images_dir, labels_dir, split='train'):
    """Spot-check image loading and dimensions."""
    print("\n" + "="*60)
    print(f"🔍 Checking image quality ({split})...")
    print("="*60)
    
    images_path = Path(images_dir) / split
    labels_path = Path(labels_dir) / split
    
    # Find images with labels
    image_files = sorted(images_path.glob('*.png')) + sorted(images_path.glob('*.jpg'))
    label_stems = set(f.stem for f in labels_path.glob('*.txt'))
    
    images_with_labels = [f for f in image_files if f.stem in label_stems and 
                         (labels_path / f"{f.stem}.txt").stat().st_size > 0]
    
    if not images_with_labels:
        print("❌ No images with labels found")
        return False
    
    print(f"   Checking {min(3, len(images_with_labels))} sample images...")
    
    errors = []
    for img_file in images_with_labels[:3]:
        try:
            img = cv2.imread(str(img_file))
            
            if img is None:
                errors.append(f"{img_file.name} - Could not load image")
                continue
            
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            
            # Read corresponding label
            label_file = labels_path / f"{img_file.stem}.txt"
            with open(label_file, 'r') as f:
                label_lines = f.readlines()
            
            print(f"   ✅ {img_file.name}: {w}x{h}, {channels} channels, {len(label_lines)} objects")
        
        except Exception as e:
            errors.append(f"{img_file.name} - {e}")
    
    if errors:
        print(f"\n   ❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"      - {error}")
        return False
    
    return True


def estimate_dataset_stats(labels_dir):
    """Estimate dataset statistics."""
    print("\n" + "="*60)
    print("📊 Dataset Statistics")
    print("="*60)
    
    stats = defaultdict(lambda: {'total': 0, 'with_objects': 0, 'object_count': 0})
    
    for split in ['train', 'val', 'test']:
        labels_path = Path(labels_dir) / split
        
        if not labels_path.exists():
            continue
        
        for label_file in labels_path.glob('*.txt'):
            stats[split]['total'] += 1
            
            if label_file.stat().st_size > 0:
                stats[split]['with_objects'] += 1
                with open(label_file, 'r') as f:
                    stats[split]['object_count'] += len(f.readlines())
    
    for split in ['train', 'val', 'test']:
        if stats[split]['total'] > 0:
            pct = (stats[split]['with_objects'] / stats[split]['total']) * 100
            print(f"\n   {split.upper()}:")
            print(f"      Total images: {stats[split]['total']}")
            print(f"      Images with objects: {stats[split]['with_objects']} ({pct:.1f}%)")
            print(f"      Total objects: {stats[split]['object_count']}")
            if stats[split]['with_objects'] > 0:
                avg = stats[split]['object_count'] / stats[split]['with_objects']
                print(f"      Avg objects per image: {avg:.2f}")


def main():
    """Run all checks."""
    print("\n" + "🔧 "*30)
    print("YOLOV8 DATASET DEBUG SCRIPT")
    print("🔧 "*30)
    
    # Paths
    yaml_path = "datasets/data.yaml"
    images_dir = "datasets/images"
    labels_dir = "datasets/labels"
    
    # Run checks
    checks = [
        ("data.yaml", check_data_yaml, [yaml_path]),
        ("train images/labels", check_images_and_labels, [images_dir, labels_dir, 'train']),
        ("val images/labels", check_images_and_labels, [images_dir, labels_dir, 'val']),
        ("test images/labels", check_images_and_labels, [images_dir, labels_dir, 'test']),
        ("train label format", check_label_format, [labels_dir, 'train']),
        ("val label format", check_label_format, [labels_dir, 'val']),
        ("train image quality", check_image_quality, [images_dir, labels_dir, 'train']),
        ("dataset stats", estimate_dataset_stats, [labels_dir]),
    ]
    
    results = []
    for check_name, check_func, args in checks:
        try:
            result = check_func(*args)
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}")
    
    print(f"\n   {passed}/{total} checks passed")
    
    if passed == total:
        print("\n   ✅ Dataset is ready for training!")
        return 0
    else:
        print("\n   ❌ Dataset has issues. Fix them before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
