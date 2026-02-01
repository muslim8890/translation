import requests
import os
import sys

url = "https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8/resolve/main/model.bin"
dest = "d:/app_file/translat/backend/models/nllb-200-ct2/model.bin"

def log(msg):
    with open("download_debug.log", "a") as f:
        f.write(str(msg) + "\n")
    print(msg)

log(f"Downloading {url} to {dest}")

try:
    if os.path.exists(dest):
        try:
            os.remove(dest)
        except: pass
        
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        log(f"Total size: {total}")
        
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (10*1024*1024) == 0:
                        log(f"Downloaded {downloaded}")
                    
    log("Done.")
    if os.path.exists(dest):
        stat = os.stat(dest)
        log(f"File size: {stat.st_size}")
        if total > 0 and stat.st_size == total:
            log("SUCCESS_VERIFIED")
        elif total == 0 and stat.st_size > 500000000:
             log("SUCCESS_ASSUMED")
        else:
             log("SIZE_MISMATCH")
    else:
        log("FILE_MISSING")
             
except Exception as e:
    log(f"Error: {e}")
