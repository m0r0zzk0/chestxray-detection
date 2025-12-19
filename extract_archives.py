"""
Extract all tar.gz archives from data/chestxray to data/images
With progress bar!
"""

import os
import tarfile
from pathlib import Path
from tqdm import tqdm

def extract_archives():
    """Extract all tar.gz archives"""
    
    source_dir = Path("data/chestxray")
    target_dir = Path("data/images")
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tar.gz files
    archives = sorted(source_dir.glob("images_*.tar.gz"))
    
    if not archives:
        print("❌ No tar.gz files found in data/chestxray/")
        return
    
    print(f"Found {len(archives)} archives")
    print(f"Extracting to: {target_dir.absolute()}\n")
    
    # Extract each archive with progress bar
    for archive_path in tqdm(archives, desc="Extracting archives"):
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=target_dir)
        except Exception as e:
            print(f"\n❌ Error extracting {archive_path.name}: {e}")
            return
    
    # Count extracted files
    png_files = list(target_dir.glob("*.png"))
    
    print(f"\n{'='*60}")
    print(f"✅ Extraction complete!")
    print(f"{'='*60}")
    print(f"Total PNG files extracted: {len(png_files)}")
    print(f"Location: {target_dir.absolute()}")
    print(f"\nExample files:")
    for f in png_files[:5]:
        print(f"  - {f.name}")
    
    return len(png_files)


if __name__ == "__main__":
    extract_archives()
