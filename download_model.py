
import os
import urllib.request
import zipfile
import shutil

def setup_model():
    print("Beginning model setup...")
    
    # Define paths
    project_root = os.getcwd()
    models_dir = os.path.join(project_root, "models")
    zip_path = os.path.join(project_root, "buffalo_l.zip")
    onnx_target_path = os.path.join(models_dir, "embedder_arcface.onnx")
    
    # 1. Create models directory
    if not os.path.exists(models_dir):
        print(f"Creating directory: {models_dir}")
        os.makedirs(models_dir)
        
    # 2. Download the file
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    # Using the GitHub release directly is often more reliable/direct than SourceForge for scripts
    # Use the one from the original repo releases if possible, or the one widely used.
    # The sourceforge one is: https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download
    # Let's try the sourceforge one again but with Python's request library which follows redirects well.
    url = "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
    
    print(f"Downloading model from {url}...")
    print("This may take a few minutes depending on your internet speed.")
    
    try:
        # Use urllib to download
        urllib.request.urlretrieve(url, zip_path)
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"Download complete. Size: {size_mb:.2f} MB")
        
        if size_mb < 100:
            print("WARNING: File seems too small. Download might have failed or downloaded an HTML page.")
            # Fallback to github release if sourceforge failed
            print("Trying fallback URL...")
            fallback_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
            urllib.request.urlretrieve(fallback_url, zip_path)
            size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            print(f"Fallback Download complete. Size: {size_mb:.2f} MB")
            
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    # 3. Extract and Move
    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List files to find the right one
            file_names = zip_ref.namelist()
            print(f"Files in zip: {file_names}")
            
            target_file_name = "w600k_r50.onnx"
            if target_file_name in file_names:
                zip_ref.extract(target_file_name, path=project_root)
                extracted_path = os.path.join(project_root, target_file_name)
                
                print(f"Moving {extracted_path} to {onnx_target_path}")
                if os.path.exists(onnx_target_path):
                   os.remove(onnx_target_path) 
                shutil.move(extracted_path, onnx_target_path)
            else:
                print(f"Could not find {target_file_name} in the zip file.")
                
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
        return
    except Exception as e:
        print(f"Error extracting/moving: {e}")
        return

    # 4. Cleanup
    print("Cleaning up...")
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        # Remove other extracted files if they exist (clean up potential mess from previous attempts)
        state_files = ["1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx"]
        for f in state_files:
            p = os.path.join(project_root, f)
            if os.path.exists(p):
                os.remove(p)
    except Exception as e:
        print(f"Warning during cleanup: {e}")

    if os.path.exists(onnx_target_path):
        print("\nSUCCESS: Model installed successfully to models/embedder_arcface.onnx")
    else:
        print("\nFAILURE: Model was not installed.")

if __name__ == "__main__":
    setup_model()
