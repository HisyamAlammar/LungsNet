import os
import shutil
import random
from glob import glob

RAW_DIR = "../data/raw/chest_xray"
PROCESSED_DIR = "../data/processed/chest_xray"

def balance_data():
    if os.path.exists(PROCESSED_DIR):
        print(f"Processed directory {PROCESSED_DIR} already exists. Deleting...")
        shutil.rmtree(PROCESSED_DIR)
    
    print(f"Creating {PROCESSED_DIR}...")
    os.makedirs(PROCESSED_DIR)

    # 1. Copy Test and Val (No balancing needed usually, but let's keep directory structure consistent)
    for subset in ['test', 'val']:
        src_path = os.path.join(RAW_DIR, subset)
        dst_path = os.path.join(PROCESSED_DIR, subset)
        if os.path.exists(src_path):
            print(f"Copying {subset} data...")
            shutil.copytree(src_path, dst_path)
    
    # 2. Balance Train Set
    train_src = os.path.join(RAW_DIR, 'train')
    train_dst = os.path.join(PROCESSED_DIR, 'train')
    os.makedirs(train_dst, exist_ok=True)

    categories = ['NORMAL', 'PNEUMONIA']
    
    # Get image paths
    normal_paths = glob(os.path.join(train_src, 'NORMAL', '*'))
    pneumonia_paths = glob(os.path.join(train_src, 'PNEUMONIA', '*'))
    
    print(f"Original Training Counts: NORMAL={len(normal_paths)}, PNEUMONIA={len(pneumonia_paths)}")
    
    # Create destination folders
    os.makedirs(os.path.join(train_dst, 'NORMAL'), exist_ok=True)
    os.makedirs(os.path.join(train_dst, 'PNEUMONIA'), exist_ok=True)
    
    # Copy Pneumonia (Majority)
    print("Copying Pneumonia images...")
    for p in pneumonia_paths:
        shutil.copy(p, os.path.join(train_dst, 'PNEUMONIA', os.path.basename(p)))
        
    # Copy Normal (Minority) + Oversample
    print("Copying and Oversampling Normal images...")
    
    # First copy originals
    for p in normal_paths:
        shutil.copy(p, os.path.join(train_dst, 'NORMAL', os.path.basename(p)))
        
    # Valid extensions filter
    valid_exts = {'.jpeg', '.jpg', '.png'}
    normal_paths = [p for p in normal_paths if os.path.splitext(p)[1].lower() in valid_exts]

    if not normal_paths:
        print("No valid normal images found to sample from!")
        return

    target_count = len(pneumonia_paths)
    current_count = len(normal_paths)
    needed = target_count - current_count
    
    if needed > 0:
        print(f"Oversampling {needed} images for NORMAL class...")
        for i in range(needed):
            src_image = random.choice(normal_paths)
            base, ext = os.path.splitext(os.path.basename(src_image))
            new_name = f"{base}_aug_{i}{ext}"
            shutil.copy(src_image, os.path.join(train_dst, 'NORMAL', new_name))
            
    print("Balancing Complete.")
    print(f"Final Training Counts: NORMAL={len(os.listdir(os.path.join(train_dst, 'NORMAL')))}, PNEUMONIA={len(os.listdir(os.path.join(train_dst, 'PNEUMONIA')))}")

if __name__ == "__main__":
    balance_data()
