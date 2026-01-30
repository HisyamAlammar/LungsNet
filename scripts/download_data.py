import os

def download_data():
    dataset_name = "paultimothymooney/chest-xray-pneumonia"
    download_dir = os.path.join("..", "data", "raw")
    
    # Ensure directory exists
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    print(f"Downloading {dataset_name} to {download_dir}...")
    
    # fast download using kaggle cli
    exit_code = os.system(f"kaggle datasets download -d {dataset_name} -p \"{download_dir}\" --unzip")
    
    if exit_code == 0:
        print("✅ Download and Unzip successful!")
    else:
        print("❌ Download failed. Please check your Kaggle API key (kaggle.json) and internet connection.")
        print("Ensure kaggle.json is in C:\\Users\\VICTUS\\.kaggle\\")

if __name__ == "__main__":
    download_data()
