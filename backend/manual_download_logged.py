import requests
import os
import traceback

LOG_FILE = "download_log.txt"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("--- Starting JustFrederik Model Test ---\n")

try:
    # Candidate 2: JustFrederik/nllb-200-distilled-600M-ct2-int8
    REPO_ID = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
    BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/"
    TEST_FILE = "config.json"
    
    log(f"Testing Repo: {REPO_ID}")
    url = BASE_URL + TEST_FILE
    
    try:
        r = requests.get(url, stream=True)
        log(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            log("SUCCESS: Repo Accessible.")
            
            # Use this repo!
            TARGET_DIR = os.path.join(os.getcwd(), "models", "nllb-200-ct2")
            os.makedirs(TARGET_DIR, exist_ok=True)
            
            FILES = ["config.json", "model.bin", "sentencepiece.bpe.model", "shared_vocabulary.json"]
            for f_name in FILES:
                 d_url = BASE_URL + f_name
                 l_path = os.path.join(TARGET_DIR, f_name)
                 log(f"Downloading {f_name}...")
                 with requests.get(d_url, stream=True) as d_r:
                     if d_r.status_code == 200:
                         with open(l_path, 'wb') as f:
                             for chunk in d_r.iter_content(chunk_size=8192):
                                 f.write(chunk)
                         log("OK")
                     else:
                         log(f"FAIL ({d_r.status_code})")
                         
        else:
             log(f"FAILURE: Repo Inaccessible ({r.status_code})")
             
    except Exception as e:
        log(f"EXCEPTION: {e}")

    log("--- Script Finished ---")

except Exception as e:
    log(f"CRITICAL SCRIPT FAILURE: {e}")
    log(traceback.format_exc())
