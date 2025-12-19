"""
Diagnostic script to check dataset structure
"""

import pandas as pd
from pathlib import Path

print("=" * 60)
print("🔍 Dataset Diagnostic")
print("=" * 60)

# 1. Check CSV
print("\n📄 CSV Analysis:")
csv_path = Path("data/annotations.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"   ✓ Found: {csv_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Unique images: {df['Image Index'].nunique()}")
    print(f"   Sample names (first 5):")
    for name in df['Image Index'].unique()[:5]:
        print(f"      - {name}")
else:
    print(f"   ✗ Missing: {csv_path}")

# 2. Check train list
print("\n📝 Train List:")
train_list = Path("data/train_val_list.txt")
if train_list.exists():
    with open(train_list) as f:
        train_names = [line.strip() for line in f if line.strip()]
    print(f"   ✓ Found: {train_list}")
    print(f"   Total names: {len(train_names)}")
    if train_names:
        print(f"   Sample names (first 5):")
        for name in train_names[:5]:
            print(f"      - {name}")
else:
    print(f"   ✗ Missing: {train_list}")

# 3. Check test list
print("\n📝 Test List:")
test_list = Path("data/test_list.txt")
if test_list.exists():
    with open(test_list) as f:
        test_names = [line.strip() for line in f if line.strip()]
    print(f"   ✓ Found: {test_list}")
    print(f"   Total names: {len(test_names)}")
    if test_names:
        print(f"   Sample names (first 5):")
        for name in test_names[:5]:
            print(f"      - {name}")
else:
    print(f"   ✗ Missing: {test_list}")

# 4. Check images directory
print("\n🖼️  Images Directory:")
img_dir = Path("data/images")
if img_dir.exists():
    img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    print(f"   ✓ Found: {img_dir}")
    print(f"   Total images: {len(img_files)}")
    if img_files:
        print(f"   Sample images (first 5):")
        for img in img_files[:5]:
            print(f"      - {img.name}")
else:
    print(f"   ✗ Missing: {img_dir}")

# 5. Check matching
print("\n🔗 Matching Analysis:")
if 'df' in locals() and train_names:
    csv_imgs = set(df['Image Index'].unique())
    train_set = set(train_names)
    test_set = set(test_names) if 'test_names' in locals() else set()
    
    matches_train = csv_imgs & train_set
    matches_test = csv_imgs & test_set
    in_neither = csv_imgs - train_set - test_set
    
    print(f"   CSV images matching train list: {len(matches_train)}")
    print(f"   CSV images matching test list: {len(matches_test)}")
    print(f"   CSV images in neither list: {len(in_neither)}")
    
    if in_neither and len(in_neither) < 10:
        print(f"   Images in neither:")
        for img in list(in_neither)[:5]:
            print(f"      - {img}")

print("\n" + "=" * 60)
