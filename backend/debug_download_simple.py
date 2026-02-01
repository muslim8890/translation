from huggingface_hub import hf_hub_download
import os

REPO_ID = "softcatala/nllb-200-distilled-600M-ct2-int8"
FILENAME = "config.json"
LOCAL_DIR = os.path.join(os.getcwd(), "models_test")

print(f"Attempting to download {FILENAME} from {REPO_ID}...")
try:
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=LOCAL_DIR)
    print(f"SUCCESS: File downloaded to {path}")
    
    # Check if file exists
    if os.path.exists(path):
        print(f"File exists at {path}")
        with open(path, 'r') as f:
            print(f"Content preview: {f.read()[:50]}...")
    else:
        print("File reported downloaded but not found!")
        
except Exception as e:
    print(f"FAILURE: {e}")
