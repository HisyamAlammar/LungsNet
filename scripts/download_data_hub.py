import kagglehub
import shutil
import os

def download_data():
    print("Downloading dataset using kagglehub...")
    
    # Download latest version
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    print("Path to dataset files:", path)
    
    # Define target directory
    target_dir = os.path.join("..", "data", "raw", "chest_xray")
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    print(f"Moving files to {target_dir}...")
    
    # Copy/Move files from cache to project directory
    # kagglehub usually downloads to %USERPROFILE%/.cache/kagglehub/...
    # Structure of this specific dataset often has a 'chest_xray' folder inside
    
    source_items = os.listdir(path)
    
    # Check if 'chest_xray' or 'chest_xray/chest_xray' exists to handle nested folders
    if 'chest_xray' in source_items:
        source_chest_xray = os.path.join(path, 'chest_xray')
        # Sometimes there's a double nesting: chest_xray/chest_xray
        if os.path.exists(os.path.join(source_chest_xray, 'chest_xray')):
             source_chest_xray = os.path.join(source_chest_xray, 'chest_xray')
    else:
        source_chest_xray = path

    # Copy tree
    if os.path.exists(target_dir):
         print(f"Target directory {target_dir} already exists. Skipping copy.")
    else:
        shutil.copytree(source_chest_xray, target_dir)
        print("✅ Data successfully moved to project folder!")

if __name__ == "__main__":
    download_data()
