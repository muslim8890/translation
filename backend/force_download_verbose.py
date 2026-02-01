import logging
import time
import os
import shutil
from huggingface_hub import snapshot_download

# Configure logging to see download progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_script")

MODEL_ID = "softcatala/nllb-200-distilled-600M-ct2-int8"
MODEL_PATH = os.path.join(os.getcwd(), "models", "nllb-200-ct2")

print(f"--- Starting Foreground Download of {MODEL_ID} ---")
print(f"Target Directory: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    print("Directory exists. Cleaning up...")
    shutil.rmtree(MODEL_PATH)
os.makedirs(MODEL_PATH, exist_ok=True)

try:
    print("Downloading snapshot...")
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_PATH, local_dir_use_symlinks=False)
    print("Snapshot download finished.")
    
    print("Checking files...")
    files = os.listdir(MODEL_PATH)
    print(f"Files found: {files}")
    
    if "sentencepiece.bpe.model" in files or any("spm" in f for f in files):
        print("SUCCESS: Tokenizer found.")
    else:
        print("FAILURE: Tokenizer missing.")

except Exception as e:
    print(f"CRITICAL FAILURE: {e}")
    import traceback
    traceback.print_exc()

print("--- Script Finished ---")
