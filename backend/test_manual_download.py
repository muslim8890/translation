import requests
import sys
import os

def log(msg):
    with open("manual_download_log.txt", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

url = "https://huggingface.co/msg/bert-base-uncased/resolve/main/config.json" # Deliberately wrong org to trigger 404? No, use correct one.
url = "https://huggingface.co/JustFrederik/nllb-200-distilled-600M-ct2-int8/resolve/main/config.json"
log(f"Downloading {url}...")


home = os.path.expanduser("~")
netrc_path = os.path.join(home, "_netrc")
log(f"Checking for _netrc at {netrc_path}: {os.path.exists(netrc_path)}")

try:
    log("Attempting with trust_env=False...")
    with requests.Session() as s:
        s.trust_env = False
        resp = s.get(url, timeout=30)
        log(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            log("Download successful with trust_env=False!")
        else:
            log(f"Failed: {resp.text[:200]}")
except Exception as e:
    log(f"Exception: {e}")


