import requests
import os
import sys
import time

url = "https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8/resolve/main/model.bin"
dest = "d:/app_file/translat/backend/models/nllb-200-ct2/model.bin"

def log(msg):
    with open("download_resume.log", "a") as f:
        f.write(str(msg) + "\n")
    print(msg)

log(f"Downloading {url} to {dest}")

def download_with_resume():
    max_retries = 20
    for attempt in range(max_retries):
        try:
            current_size = 0
            if os.path.exists(dest):
                current_size = os.path.getsize(dest)
            
            headers = {}
            if current_size > 0:
                headers['Range'] = f'bytes={current_size}-'
                log(f"Resuming from {current_size} bytes...")
            
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                # 416 means range not satisfiable (file complete?)
                if r.status_code == 416: 
                    log("Range not satisfiable (completed?)")
                    return True

                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                if current_size == 0:
                    log(f"Total size to download: {total}")
                
                 # Note: content-length with Range is just the REMAINING part
                
                downloaded = current_size
                with open(dest, 'ab') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): 
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (10*1024*1024) == 0:
                                log(f"Downloaded {downloaded}")
                
            log("Download finished cleanly (loop end).")
            return True

        except Exception as e:
            log(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5)
            
    return False

if download_with_resume():
    if os.path.exists(dest):
        log(f"Final Size: {os.path.getsize(dest)}")
        # Verify ~622MB
        if os.path.getsize(dest) > 600000000:
             log("SUCCESS_VERIFIED")
        else:
             log("SIZE_MISMATCH")
else:
    log("FAILED_AFTER_RETRIES")
