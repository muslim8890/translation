import requests
import os
import sys

def log(msg):
    with open("manual_model_download.log", "a") as f:
        f.write(str(msg) + "\n")
    print(msg)

url = "https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8/resolve/main/model.bin"
dest = "d:/app_file/translat/backend/models/nllb-200-ct2/model.bin"

log(f"Downloading {url} to {dest}...")

try:
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (1024*1024*10) == 0: # Log every 10MB
                    print(f"Downloaded {downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB")
        
    log("Download of model.bin complete.")
except Exception as e:
    log(f"Failed: {e}")
