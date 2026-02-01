import requests
import os
import shutil

BASE_URL = "https://huggingface.co/softcatala/nllb-200-distilled-600M-ct2-int8/resolve/main/"
FILES_TO_DOWNLOAD = [
    "config.json",
    "model.bin",
    "sentencepiece.bpe.model",
    "shared_vocabulary.json"
]

TARGET_DIR = os.path.join(os.getcwd(), "models", "nllb-200-ct2")
os.makedirs(TARGET_DIR, exist_ok=True)

print(f"Starting direct download to {TARGET_DIR}...")

for filename in FILES_TO_DOWNLOAD:
    url = BASE_URL + filename
    local_path = os.path.join(TARGET_DIR, filename)
    
    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"SUCCESS: {filename}")
    except Exception as e:
        print(f"FAILED: {filename} ({e})")

print("--- Manual Download Complete ---")
