import urllib.request
import os
import traceback

LOG_FILE = "download_log_urllib.txt"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("--- Starting Urllib Download ---\n")

try:
    REPO_ID = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
    BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/"
    
    TARGET_DIR = os.path.join(os.getcwd(), "models", "nllb-200-ct2")
    os.makedirs(TARGET_DIR, exist_ok=True)
    log(f"Target: {TARGET_DIR}")

    # Prioritize SentencePiece (Critical & Small)
    FILES = ["sentencepiece.bpe.model", "config.json", "shared_vocabulary.json"]
    
    for filename in FILES:
        url = BASE_URL + filename
        local_path = os.path.join(TARGET_DIR, filename)
        log(f"Downloading {filename}...")
        
        try:
            urllib.request.urlretrieve(url, local_path)
            size = os.path.getsize(local_path)
            log(f"SUCCESS: {filename} ({size} bytes)")
        except Exception as e:
            log(f"FAILED: {filename} ({e})")

    log("--- Critical Files Done ---")
    
    # Try model.bin last
    filename = "model.bin"
    url = BASE_URL + filename
    local_path = os.path.join(TARGET_DIR, filename)
    log(f"Downloading {filename} (Large)...")
    try:
        urllib.request.urlretrieve(url, local_path)
        size = os.path.getsize(local_path)
        log(f"SUCCESS: {filename} ({size} bytes)")
    except Exception as e:
        log(f"FAILED: {filename} ({e})")

except Exception as e:
    log(f"CRITICAL SCRIPT FAILURE: {e}")
    log(traceback.format_exc())
